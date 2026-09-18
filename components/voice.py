import io
import json
import os
import subprocess
import wave

# Task ids the robot can run. Phrases are not listed; the LLM maps free speech.
TASKS = {
    "home": "Move the arm to the taught home pose.",
    "sort": "Pick objects and place them in a bin. Extract each object color and its destination separately.",
    "locate": "Go home, detect blocks, print world XY and pick Z. Do not grasp.",
    "capture": "Capture a color and depth frame from the camera and print block depths.",
    "quit": "Stop the voice loop and exit.",
}

VALID_COLORS = ("red", "yellow")
VALID_PLACES = ("bin1", "bin2")
DEFAULT_MOVES = [{"color": "red", "place": "bin1"}, {"color": "yellow", "place": "bin2"}]

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.6-luna")


def speak(text: str) -> None:
    print(text)
    try:
        subprocess.run(["say", text], check=False)
    except Exception:
        pass


def listen_dictation(prompt: str = "> ") -> str:
    """Type or use Mac Dictation. Do not use Globe/Fn — that opens emoji.

    System Settings → Keyboard → Dictation → Shortcut → Press Control Key Twice.
    Then Control-Control, speak, Enter.
    """
    print(
        "Dictation: press Control twice (not Globe/Fn — that is emoji), speak, Enter.\n"
        "Or just type the request."
    )
    try:
        return input(prompt).strip()
    except EOFError:
        return "quit"


def transcribe_audio(data: bytes, filename: str = "speech.webm") -> str:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    buf = io.BytesIO(data)
    buf.name = filename
    text = OpenAI().audio.transcriptions.create(model="whisper-1", file=buf).text
    return (text or "").strip()


def listen_mic(seconds: float | None = None) -> str:
    """Record from the Mac mic, then transcribe with Whisper."""
    import numpy as np
    import sounddevice as sd

    seconds = float(os.environ.get("VOICE_LISTEN_S", seconds or 5))
    rate = 16000
    speak(f"Listening for {seconds:.0f} seconds.")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(audio).tobytes())
    return transcribe_audio(buf.getvalue(), "speech.wav")


def listen(mode: str = "mic") -> str:
    if mode == "dictation":
        return listen_dictation()
    return listen_mic()


def _normalize_place(value: str) -> str | None:
    p = "".join(ch for ch in str(value).lower() if ch.isalnum())
    if p in ("bin1", "b1", "1"):
        return "bin1"
    if p in ("bin2", "b2", "2"):
        return "bin2"
    return None


def _normalize_color(value: str) -> str | None:
    c = str(value).lower().strip()
    return c if c in VALID_COLORS else None


def moves_to_bins(moves: list) -> dict:
    bins = {}
    for move in moves:
        color = _normalize_color(move.get("color", ""))
        place = _normalize_place(move.get("place", ""))
        if color and place:
            bins[color] = place
    return bins


def map_task(text: str) -> dict:
    """Ask the LLM for the task, plus pick color and place target when sorting."""
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it to .env (same key as the other cell)."
        )
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in TASKS.items())
    client = OpenAI()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You map a spoken request to one robot task. "
                    "Reply with JSON only:\n"
                    '{"task":"<id>","say":"<short confirmation>",'
                    '"moves":[{"color":"<color>","place":"<bin>"}]}\n'
                    "Rules:\n"
                    "- task must be one of the ids below, or unknown.\n"
                    "- Extract object color and destination separately. "
                    "Do not assume red always goes to bin 1.\n"
                    "- color must be one of: red, yellow.\n"
                    "- place must be bin1 or bin2.\n"
                    "- One spoken assignment is one move. "
                    '"red blocks to bin 2" → '
                    '[{"color":"red","place":"bin2"}].\n'
                    '- "put red in bin 2 and yellow in bin 1" → two moves.\n'
                    "- If they say sort / pick and place but do not name "
                    "colors or bins, use "
                    '[{"color":"red","place":"bin1"},'
                    '{"color":"yellow","place":"bin2"}].\n'
                    "- For home, locate, capture, quit, use moves: [].\n"
                    "- Do not invent tasks.\n\n"
                    f"Tasks:\n{catalog}"
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"task": "unknown", "say": "I could not parse that.", "moves": []}
    task = str(data.get("task", "unknown")).strip().lower()
    if task not in TASKS and task != "unknown":
        task = "unknown"
    moves = data.get("moves") if isinstance(data.get("moves"), list) else []
    if task == "sort" and not moves_to_bins(moves):
        moves = list(DEFAULT_MOVES)
    say = str(data.get("say") or "").strip() or (
        "Okay." if task in TASKS else "I am not sure what you want."
    )
    return {"task": task, "say": say, "moves": moves}
