"""TranscriptionSync — real-time, training-free, CPU-cheap lip-sync from a
transcription text stream to ARKit mouth blendshapes.

    from transcription_sync import VisemeStream

    vs = VisemeStream(fps=60)
    for chunk in llm_transcription_stream:      # any LLM / ASR text stream
        vs.feed_text(chunk)
    for frame in vs.frames():                   # {arkit_channel: value}
        send_to_avatar(frame)

See the paper: "TranscriptionSync: Phoneme-Aware Real-Time Lip Synchronization
for Native-Audio Large Language Models".
"""
from .amplitude import EnvelopeFollower, rms_envelope
from .engine import VisemeStream
from .visemes import CHANNELS, VISEME, WORD_PAUSE

__version__ = "0.1.0"
__all__ = [
    "VisemeStream",
    "rms_envelope",
    "EnvelopeFollower",
    "CHANNELS",
    "VISEME",
    "WORD_PAUSE",
    "__version__",
]
