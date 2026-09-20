"""Dead-man watchdog timer for velocity / servo streaming.

Velocity and servo-streaming control are only safe while the client keeps
sending fresh commands. `Watchdog` is a small background timer: every call to
`feed()` (re)arms it for `timeout_s` seconds; if `timeout_s` elapses with no
further `feed()`, `on_timeout` is invoked from a background thread. The arm
component uses this to zero velocity / stop the arm the moment a client goes
quiet (crashes, disconnects, stalls, etc.).

Implemented with `threading.Timer` rather than asyncio so it keeps working
regardless of what the calling event loop is doing, and so it is trivial to
unit test without an event loop.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional


class Watchdog:
    def __init__(self, timeout_s: float, on_timeout: Callable[[], None]):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self.timeout_s = timeout_s
        self._on_timeout = on_timeout
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._armed = False
        self.timeout_count = 0

    def feed(self) -> None:
        """(Re)arm the watchdog. Call this on every fresh velocity/servo command."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._armed = True
            timer = threading.Timer(self.timeout_s, self._fire)
            timer.daemon = True
            self._timer = timer
            timer.start()

    def stop(self) -> None:
        """Disarm the watchdog. No timeout will fire until `feed()` is called again."""
        with self._lock:
            self._armed = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def _fire(self) -> None:
        with self._lock:
            if not self._armed:
                return
            self._armed = False
            self._timer = None
        self.timeout_count += 1
        self._on_timeout()
