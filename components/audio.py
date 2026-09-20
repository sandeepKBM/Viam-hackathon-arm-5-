"""Audio I/O -- STUB.

The hand-over flow (components/handover.py) talks to the person: it announces
what it's doing and ASKS whether they're ready before releasing the object.
This module is the seam for that. It is deliberately a STUB -- swap the two
methods for real text-to-speech / speech-to-text when you wire audio hardware:

  - speak(text):            TODO real TTS -- e.g. pyttsx3, a cloud TTS, or a
                            Viam audio_output/speech service on the machine.
  - ask_yes_no(question):   TODO real STT -- capture the mic (a Viam
                            audio_input component), transcribe (e.g. Whisper),
                            and parse a yes/no. Right now it returns `default`
                            (or a scripted answer via ScriptedAudio) so the
                            rest of the pipeline runs end-to-end offline.

Nothing here blocks on real hardware, so the pick/hand-over loop is fully
runnable (and testable) before the audio stack exists.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional


class AudioIO:
    """Base stub. Prints instead of speaking, and returns a default answer
    instead of listening. Subclass / replace the two methods for real audio."""

    def speak(self, text: str) -> None:
        # TODO: real TTS. For now, surface it on the console.
        print(f"[assistant \U0001F50A] {text}")

    def ask_yes_no(self, question: str, *, default: bool = False) -> bool:
        # TODO: real STT -- record from the mic, transcribe, parse yes/no.
        # Stub: announce the question and fall back to `default` so an
        # unattended/offline run doesn't hang waiting for a voice.
        print(f"[assistant \U0001F50A?] {question}  (stub -> {'yes' if default else 'no'})")
        return default


class ScriptedAudio(AudioIO):
    """Test/demo double: `ask_yes_no` returns queued answers in order (then
    falls back to `default`). Lets the hand-over flow be exercised offline
    with a deterministic 'the person said yes' / '...said no'."""

    def __init__(self, answers: Optional[list] = None) -> None:
        self._answers: Deque[bool] = deque(bool(a) for a in (answers or []))
        self.said: list = []          # log of everything spoken (for asserts)
        self.asked: list = []          # log of questions asked

    def speak(self, text: str) -> None:
        self.said.append(text)

    def ask_yes_no(self, question: str, *, default: bool = False) -> bool:
        self.asked.append(question)
        return self._answers.popleft() if self._answers else default
