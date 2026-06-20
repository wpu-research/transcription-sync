"""VisemeStream — streaming text -> ARKit blendshape frames.

This is the real-time, training-free, CPU-cheap lip-sync core of TranscriptionSync.
It consumes a transcription text stream (from any LLM / ASR — the text arrives
*before or with* the audio, so the mouth is ready without buffering the utterance)
and produces smoothed 14-channel ARKit mouth poses at a fixed character cadence.

Design (ported faithfully from the deployed A.T.L.A.S engine):
  - each character maps to an articulatory viseme pose (see visemes.py);
  - poses are scaled by a live amplitude envelope so silence keeps the mouth shut;
  - an exponential smoother (alpha) interpolates between successive poses;
  - inter-word characters render a brief labial closure.

Two usage modes:
  - Pull/offline:  render_text(text) -> list[frame]   (deterministic, for tests)
  - Push/real-time: feed_text(chunk); repeatedly call tick() at the char cadence,
    or use drive_realtime() which paces tick() with time.sleep.

A "frame" is a dict {arkit_channel: value in [0,1]} over the 14 mouth channels.
"""
from __future__ import annotations

import time
from typing import Callable, Iterator, Optional

from .visemes import CHANNELS, PAUSE_CHARS, VISEME, WORD_PAUSE


class VisemeStream:
    def __init__(
        self,
        fps: int = 60,
        chars_per_sec: float = 13.0,
        smoothing: float = 0.40,
        amplitude_gain: float = 2.2,
        amplitude_max: float = 1.4,
    ):
        """
        fps            : nominal output frame rate (metadata; cadence is char-driven).
        chars_per_sec  : viseme advance rate (~150 wpm x 5 chars / 60 s = 13).
        smoothing      : per-tick interpolation factor toward the target pose.
        amplitude_gain : envelope -> pose-scale multiplier.
        amplitude_max  : clamp on the scale factor.
        """
        self.fps = fps
        self.chars_per_sec = chars_per_sec
        self.char_dur = 1.0 / chars_per_sec
        self.smoothing = smoothing
        self.amplitude_gain = amplitude_gain
        self.amplitude_max = amplitude_max

        self._current = {k: 0.0 for k in CHANNELS}
        self._pending: list[tuple[str, str]] = []
        self._amp = 0.0

    # ── input ────────────────────────────────────────────────────────────────
    def set_amplitude(self, envelope: float) -> None:
        """Set the current speech amplitude envelope (e.g. RMS in ~[0, 0.42]).
        Scales pose magnitude so the mouth stays shut during silence."""
        self._amp = max(0.0, float(envelope))

    def feed_text(self, text: str) -> None:
        """Enqueue a transcription chunk; letters become visemes, punctuation/space
        become inter-word pauses. Safe to call repeatedly as the stream arrives."""
        for c in text.lower():
            if c.isalpha():
                self._pending.append(("char", c))
            elif c in PAUSE_CHARS:
                self._pending.append(("pause", c))

    @property
    def pending(self) -> int:
        """Number of queued characters not yet rendered."""
        return len(self._pending)

    def reset(self) -> None:
        """Clear queue and relax the mouth to neutral (e.g. at end of a turn)."""
        self._pending.clear()
        self._current = {k: 0.0 for k in CHANNELS}
        self._amp = 0.0

    # ── output ───────────────────────────────────────────────────────────────
    def tick(self) -> Optional[dict[str, float]]:
        """Advance one character and return the smoothed blendshape frame.
        Returns None if no characters are pending."""
        if not self._pending:
            return None
        kind, ch = self._pending.pop(0)
        target = WORD_PAUSE if kind == "pause" else VISEME.get(ch, VISEME["_"])

        amp = min(self.amplitude_max, self._amp * self.amplitude_gain)
        scaled = {k: min(1.0, v * amp) for k, v in target.items()}
        # Keep lips slightly closed unless the jaw is actively open.
        if "MouthClose" not in scaled:
            scaled["MouthClose"] = max(0.0, 0.08 - scaled.get("JawOpen", 0.0) * 0.2)

        a = self.smoothing
        for k in CHANNELS:
            self._current[k] += a * (scaled.get(k, 0.0) - self._current[k])
        return dict(self._current)

    def render_text(
        self, text: str, amplitude: float = 0.2
    ) -> list[dict[str, float]]:
        """Offline: render a full string to a list of frames at a constant
        amplitude. Deterministic — useful for tests and previews."""
        self.feed_text(text)
        self.set_amplitude(amplitude)
        frames = []
        while self._pending:
            f = self.tick()
            if f is not None:
                frames.append(f)
        return frames

    def drive_realtime(
        self,
        on_frame: Callable[[dict[str, float]], None],
        stop: Optional[Callable[[], bool]] = None,
        idle_sleep: float = 0.02,
    ) -> None:
        """Blocking real-time driver: paces tick() at the character cadence and
        invokes on_frame(frame) for each. Feed text from another thread via
        feed_text()/set_amplitude(). Runs until stop() returns True (if given)
        and the queue is drained."""
        while True:
            if not self._pending:
                if stop is not None and stop():
                    return
                time.sleep(idle_sleep)
                continue
            frame = self.tick()
            if frame is not None:
                on_frame(frame)
            time.sleep(self.char_dur)

    def frames(self) -> Iterator[dict[str, float]]:
        """Generator yielding currently-pending frames (non-blocking, no pacing)."""
        while self._pending:
            f = self.tick()
            if f is not None:
                yield f
