"""Timeout helpers for CLI operations."""

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import Iterator


def supports_signal_timeout() -> bool:
    """Return True when SIGALRM-based timeout is available."""
    return hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")


@contextmanager
def timeout_guard(seconds: float) -> Iterator[None]:
    """Raise TimeoutError if execution exceeds ``seconds``.

    A timeout of 0 disables the guard.
    """
    if seconds <= 0 or not supports_signal_timeout():
        yield
        return

    def handle_timeout(signum, frame):
        raise TimeoutError(f"timed out after {seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
