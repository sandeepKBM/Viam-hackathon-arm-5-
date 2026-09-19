import io
import json
import os
import re
import subprocess
import wave

# Task ids the robot can run. Phrases are not listed; the LLM maps free speech.
TASKS = {
    "home": "Move the arm to the taught home pose.",
    "dropoff": "Move the arm to drop-off. Use when they say drop off / drop it off and do not name an object.",
    "handoff": "Move the arm to the taught hand-off. Use when they say give it to him / me / her, hand it over, and do not name an object.",
    "sort": "Pick a named object and take it to a destination. Extract the object and its destination separately.",
    "locate": "Go home, detect pick objects, print world XY and pick Z. Do not grasp.",
    "capture": "Capture a color and depth frame from the camera and print object depths.",
    "quit": "Stop the voice loop and exit.",
}

VALID_OBJECTS = ("red", "yellow", "can", "cup", "airpods", "pen", "bottle")
VALID_PLACES = ("bin1", "bin2", "dropoff", "handoff")
VALID_GOALS = ("hydrate", "write")
GENERIC_SORT = re.compile(
    r"\b(sort the blocks|sort blocks|pick and place|sort(?: them)?)\b"
)
DEFAULT_MOVES = [
    {"object": "red", "place": "bin1", "count": None},
    {"object": "yellow", "place": "bin2", "count": None},
]
COUNT_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
OBJECT_PHRASES = {
    "yellow": ("yellow blocks", "yellow block", "yellow"),
    "red": ("red blocks", "red block", "red"),
    "can": ("soda cans", "soda can", "cans", "can"),
    "cup": ("cups", "cup", "mugs", "mug"),
    "airpods": ("airpods", "airpod", "earbuds"),
    "pen": ("pens", "pen"),
    "bottle": ("bottles", "bottle"),
}

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
    from components.constants import normalize_place

    return normalize_place(value)


def _normalize_object(value: str) -> str | None:
    from components.constants import normalize_object

    return normalize_object(value)


def _place_from_speech(text: str) -> str | None:
    raw = " ".join(str(text).lower().split())
    compact = "".join(ch for ch in raw if ch.isalnum())
    give = (
        "giveittohim",
        "giveittome",
        "giveittoher",
        "givetohim",
        "givetome",
        "givetoher",
        "givehim",
        "giveme",
        "giveher",
        "handitover",
        "handittohim",
        "handittome",
        "handme",
        "handoff",
        "handover",
    )
    if (
        any(p in compact for p in give)
        or "give it to" in raw
        or "hand it" in raw
        or raw.startswith("hand me")
        or " hand me " in f" {raw} "
    ):
        return "handoff"
    if "dropoff" in compact or "drop it off" in raw or "drop off" in raw:
        return "dropoff"
    return None


def _count_token(token: str) -> int | None:
    raw = str(token).strip().lower()
    if raw in {"all", "every", "each", "null", "none", "*"}:
        return None
    if raw.isdigit():
        n = int(raw)
        return n if n > 0 else 1
    return COUNT_WORDS.get(raw)


def _parse_count(value) -> int | None:
    if value is None:
        return 1
    return _count_token(value)


def _counts_from_speech(text: str) -> dict:
    raw = " ".join(str(text).lower().split())
    wants_all = bool(re.search(r"\b(all|every|each)\b", raw))
    words = "|".join(re.escape(w) for w in COUNT_WORDS)
    counts: dict = {}
    for obj, phrases in OBJECT_PHRASES.items():
        hit = next((p for p in phrases if re.search(rf"\b{re.escape(p)}\b", raw)), None)
        if not hit:
            continue
        if wants_all:
            counts[obj] = None
            continue
        match = re.search(rf"\b(\d+|{words})\s+{re.escape(hit)}\b", raw)
        if match:
            counts[obj] = _count_token(match.group(1))
            continue
        counts[obj] = None if ("blocks" in hit or hit.endswith("s")) else 1
    return counts


def moves_to_bins(moves: list) -> dict:
    bins, _counts = moves_to_plan(moves)
    return bins


def _normalize_goal(value) -> str | None:
    raw = str(value or "").strip().lower()
    if raw in {"", "null", "none"}:
        return None
    return raw if raw in VALID_GOALS else None


def _explicit_objects(text: str) -> set[str]:
    raw = " ".join(str(text).lower().split())
    found = set()
    for obj, phrases in OBJECT_PHRASES.items():
        if any(re.search(rf"\b{re.escape(p)}\b", raw) for p in phrases):
            found.add(obj)
    return found


def _goal_from_speech(text: str) -> str | None:
    raw = " ".join(str(text).lower().split())
    if re.search(r"\b(thirsty|need a drink|want a drink|get a drink)\b", raw):
        return "hydrate"
    if re.search(r"\b(want to write|need to write|take notes|need a pen)\b", raw):
        return "write"
    return None


def _is_generic_sort(text: str) -> bool:
    raw = " ".join(str(text).lower().split())
    return bool(GENERIC_SORT.search(raw)) and not _explicit_objects(text)


def _intent_prompt() -> str:
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in TASKS.items())
    return (
        "You extract spoken intent for a robot arm. "
        "Do not claim that an object is present. "
        "Explicit commands always take priority over inferred goals. "
        "Never invent a pour. There is no pouring primitive.\n"
        "Reply with JSON only:\n"
        '{"task":"sort|home|dropoff|handoff|locate|capture|quit|unknown",'
        '"goal":"hydrate|write|null",'
        '"say":"short acknowledgement",'
        '"moves":[{"object":"...","place":"...","count":1}]}\n'
        "Rules:\n"
        "- Extract intent. Do not assert that a bottle, pen, or cup is in view.\n"
        "- task must be one of the ids below, or unknown.\n"
        "- goal is hydrate, write, or null.\n"
        '- "I am thirsty" / "I need a drink" → goal=hydrate, task=sort, moves=[].\n'
        '- "I want to write" / "I need to take notes" → goal=write, task=sort, moves=[].\n'
        "- Hydrate means hand over one bottle later, if vision finds a safe one. "
        "Do not name bottle in moves unless they said bottle.\n"
        "- Write means hand over one pen later, if vision finds a safe one. "
        "Do not name pen in moves unless they said pen.\n"
        "- Do not pour. Do not mention pouring or filling a cup.\n"
        "- object must be one of: red, yellow, can, cup, airpods, pen, bottle.\n"
        "- place must be one of: bin1, bin2, dropoff, handoff.\n"
        "- count is a positive integer, or null for all of that object.\n"
        "- If they name an object and a destination, that explicit move wins "
        "and goal is null.\n"
        '- "put the pen in bin 2" → task sort, goal null, '
        '[{"object":"pen","place":"bin2","count":1}].\n'
        '- "hand me two bottles" → task sort, goal null, '
        '[{"object":"bottle","place":"handoff","count":2}].\n'
        '- "hand me a yellow block" → task sort, '
        '[{"object":"yellow","place":"handoff","count":1}].\n'
        '- "give it to him" with no object → task handoff, goal null, moves [].\n'
        '- "drop it off" with no object → task dropoff, goal null, moves [].\n'
        '- "sort the blocks" / generic pick and place with no object → task sort, '
        "goal null, moves []. Defaults are applied later only for that case.\n"
        "- Unsupported or ambiguous requests: task=unknown, goal=null, moves=[], "
        "and ask one short clarifying question.\n"
        "- For home, dropoff, handoff, locate, capture, quit, use moves [] and goal null.\n"
        "- say is a short acknowledgement, not a success report.\n"
        "- Do not invent tasks.\n\n"
        f"Tasks:\n{catalog}"
    )


def _finalize_mapped(text: str, data: dict) -> dict:
    task = str(data.get("task", "unknown")).strip().lower()
    if task not in TASKS and task != "unknown":
        task = "unknown"
    goal = _normalize_goal(data.get("goal")) or _goal_from_speech(text)
    raw_moves = data.get("moves") if isinstance(data.get("moves"), list) else []
    spoken_counts = _counts_from_speech(text)
    spoken_objects = _explicit_objects(text)
    moves = []
    for move in raw_moves:
        if not isinstance(move, dict):
            continue
        obj = _normalize_object(move.get("object") or move.get("color") or "")
        place = _normalize_place(move.get("place", ""))
        if not obj or not place:
            continue
        if obj in spoken_counts:
            count = spoken_counts[obj]
        elif "count" in move:
            count = None if move.get("count") is None else _parse_count(move.get("count"))
        else:
            count = 1
        moves.append({"object": obj, "place": place, "count": count})
    spoken_place = _place_from_speech(text)
    if spoken_place and moves:
        moves = [
            {"object": m["object"], "place": spoken_place, "count": m.get("count", 1)}
            for m in moves
        ]
        task = "sort"
        goal = None
    elif spoken_place and spoken_objects:
        task = "sort"
        goal = None
        if not moves:
            moves = [
                {
                    "object": obj,
                    "place": spoken_place,
                    "count": spoken_counts.get(obj, 1),
                }
                for obj in spoken_objects
            ]
    elif spoken_place and task in {"sort", "unknown", "dropoff", "handoff"} and not goal:
        task = spoken_place
        moves = []
    if goal and not spoken_objects:
        moves = []
        if task in {"unknown", "sort", "handoff", "dropoff"}:
            task = "sort"
    elif spoken_objects:
        goal = None
    if task == "sort" and not moves and not goal:
        if _is_generic_sort(text):
            moves = [dict(m) for m in DEFAULT_MOVES]
        else:
            task = "unknown"
    say = str(data.get("say") or "").strip()
    if task == "unknown":
        say = say if say.endswith("?") else "What should I pick up?"
    elif not say:
        say = "Okay." if task in TASKS else "I am not sure what you want."
    return {"task": task, "goal": goal, "say": say, "moves": moves}


def moves_to_plan(moves: list) -> tuple[dict, dict]:
    bins: dict = {}
    counts: dict = {}
    for move in moves:
        obj = _normalize_object(move.get("object") or move.get("color") or "")
        place = _normalize_place(move.get("place", ""))
        if not obj or not place:
            continue
        bins[obj] = place
        counts[obj] = move["count"] if "count" in move else 1
    return bins, counts


def map_task(text: str) -> dict:
    """Ask the LLM for the task, plus pick object and place target when sorting."""
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Add it to .env (same key as the other cell)."
        )
    client = OpenAI()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _intent_prompt()},
            {"role": "user", "content": text},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "task": "unknown",
            "goal": None,
            "say": "I could not parse that. What should I pick up?",
            "moves": [],
        }
    if not isinstance(data, dict):
        data = {}
    return _finalize_mapped(text, data)
