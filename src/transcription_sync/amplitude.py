"""Audio amplitude envelope helper.

Optional companion to VisemeStream: turns a chunk of PCM audio into an RMS
envelope value to feed into VisemeStream.set_amplitude(), so viseme magnitude
tracks loudness and the mouth stays shut during silence.

numpy is only needed to actually compute an envelope; it is imported lazily so
the core engine (`import transcription_sync`) stays dependency-free. Install with
the optional extra to use these helpers:  pip install transcription-sync[audio]
"""
from __future__ import annotations


def _require_numpy():
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "transcription_sync.amplitude needs numpy. "
            "Install it with:  pip install transcription-sync[audio]"
        ) from e
    return np


def rms_envelope(
    pcm_bytes: bytes,
    *,
    dtype: str = "int16",
    full_scale: float = 32768.0,
    gamma: float = 0.55,
    gain: float = 1.9,
    cap: float = 0.42,
) -> float:
    """Return a smoothed-magnitude envelope value in [0, cap] from raw PCM.

    pcm_bytes  : little-endian PCM samples (mono).
    gamma/gain : perceptual compression and scaling of the RMS.
    cap        : upper clamp (matches the deployed engine's 0.42).
    """
    np = _require_numpy()
    samples = np.frombuffer(pcm_bytes, dtype=dtype).astype(np.float32) / full_scale
    if samples.size < 16:
        return 0.0
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return min(cap, (rms ** gamma) * gain)


class EnvelopeFollower:
    """Asymmetric (fast-attack, slow-release) smoother over successive envelope
    values, matching the deployed engine's attack=0.50 / release=0.18.

    Pure arithmetic — does not require numpy."""

    def __init__(self, attack: float = 0.50, release: float = 0.18):
        self.attack = attack
        self.release = release
        self.value = 0.0

    def update(self, raw: float) -> float:
        a = self.attack if raw > self.value else self.release
        self.value += a * (raw - self.value)
        return self.value
