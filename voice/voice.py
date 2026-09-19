"""Voice control UI: hold the button, speak, LLM maps to a robot task.

  python voice/voice.py              # opens http://127.0.0.1:8766
  python voice/voice.py --cli        # old Enter-to-record CLI
  python voice/voice.py --dictation  # Control twice, then type/dictate

Requires OPENAI_API_KEY in .env.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

from components.voice import map_task, moves_to_plan, speak, transcribe_audio

STATIC = HERE / "static"
HOST = os.environ.get("VOICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("VOICE_PORT", "8766"))

app = FastAPI()
_state = {"busy": False, "last": {}}


async def run_task(name: str, mapped: dict | None = None) -> None:
    if name == "home":
        from go_home import main

        await main()
        return
    if name == "dropoff":
        from components.arm import ArmComponent
        from components.connection import connect_machine
        from components.gripper import GripperComponent

        machine = await connect_machine()
        try:
            arm = ArmComponent(machine)
            gripper = GripperComponent(machine)
            print("Moving to dropoff...")
            await arm.go_to("dropoff")
            await gripper.open_full()
        finally:
            await machine.close()
        return
    if name == "handoff":
        from components.arm import ArmComponent
        from components.connection import connect_machine
        from components.gripper import GripperComponent
        from components.pickplace import PickPlace

        machine = await connect_machine()
        try:
            print("Moving to handoff...")
            await PickPlace(ArmComponent(machine), GripperComponent(machine)).hand_to_human()
        finally:
            await machine.close()
        return
    if name == "sort":
        from sort_blocks import main

        bins, counts = moves_to_plan((mapped or {}).get("moves") or [])
        await main(bins or None, counts or None)
        return
    if name == "locate":
        from locate_blocks import main

        await main()
        return
    if name == "capture":
        from capture_image import main

        await main()
        return
    raise ValueError(name)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/status")
async def status() -> dict:
    return {"busy": _state["busy"], "last": _state["last"]}


@app.post("/api/talk")
async def talk(audio: UploadFile = File(...)) -> dict:
    if _state["busy"]:
        return {"ok": False, "error": "arm is still moving"}
    data = await audio.read()
    if not data:
        return {"ok": False, "error": "no audio"}
    try:
        heard = transcribe_audio(data, audio.filename or "speech.webm")
        print(f"heard: {heard!r}")
        if not heard:
            return {"ok": True, "heard": "", "task": "unknown", "say": "I did not hear anything.", "moves": []}
        mapped = map_task(heard)
    except Exception as exc:
        print(f"talk failed: {exc}")
        return {"ok": False, "error": str(exc)}
    print(f"llm: {mapped}")
    speak(mapped["say"])
    payload = {"ok": True, "heard": heard, **mapped}
    _state["last"] = payload
    task = mapped["task"]
    if task in {"unknown", "quit"}:
        return payload

    async def _run() -> None:
        _state["busy"] = True
        try:
            await run_task(task, mapped)
        except Exception as exc:
            print(f"task failed: {exc}")
            speak("That task failed.")
        finally:
            _state["busy"] = False
            speak("Ready for a task.")

    asyncio.create_task(_run())
    return payload


async def session(mode: str) -> None:
    from components.voice import listen

    speak("Ready for a task.")
    while True:
        if mode == "mic":
            try:
                input("Press Enter, then speak.\n")
            except EOFError:
                return
        raw = listen(mode=mode)
        if not raw:
            speak("I did not hear anything.")
            continue
        print(f"heard: {raw!r}")
        mapped = map_task(raw)
        print(f"llm: {mapped}")
        speak(mapped["say"])
        if mapped["task"] == "unknown":
            continue
        if mapped["task"] == "quit":
            return
        await run_task(mapped["task"], mapped)
        speak("Ready for a task.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice tasks")
    parser.add_argument("--cli", action="store_true", help="terminal mic instead of the UI")
    parser.add_argument(
        "--dictation",
        action="store_true",
        help="use Mac Dictation into the prompt (Control twice, not Globe/Fn)",
    )
    args = parser.parse_args()
    if args.cli or args.dictation:
        asyncio.run(session("dictation" if args.dictation else "mic"))
        return
    import uvicorn

    print(f"Voice UI: http://{HOST}:{PORT}")
    try:
        from components.sam import load_sam

        load_sam()
    except Exception as exc:
        print(f"SAM preload failed: {exc}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
