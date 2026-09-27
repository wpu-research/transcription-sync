# TranscriptionSync

**Real-time, training-free, CPU-cheap lip-sync** that turns a *transcription text
stream* into [ARKit](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation)
mouth blendshapes — no GPU, no phoneme-timing metadata, no per-utterance buffering.

Native-audio LLMs (Gemini Live, GPT-4o Realtime) emit raw PCM with **no phoneme
timing**, which breaks the classic lip-sync toolchain (forced aligners are
non-causal; learned audio-driven models need a GPU). TranscriptionSync drives the
mouth from the model's *output transcription* instead, which arrives **before or
with** the audio — so the mouth is ready without waiting for the utterance to finish.

This package is the streaming viseme core of the method described in
*"TranscriptionSync: Phoneme-Aware Real-Time Lip Synchronization for Native-Audio
Large Language Models"* and deployed in the A.T.L.A.S desktop assistant.

## Why

Measured added latency to first mouth motion, same CPU, 50 utterances:

| Engine | Architecture | Added latency |
|---|---|---|
| **TranscriptionSync (this)** | streaming text→viseme | mouth ready **~349 ms before** audio |
| Learned audio-driven (HuBERT, 326 M) | buffered (whole utterance) | ~21 s (CPU) |
| Forced-alignment oracle (MFA) | buffered (utterance + transcript) | ~17.6 s |

The streaming design adds **no per-utterance buffer and no neural inference** —
its per-frame cost is a dictionary lookup (~0.8 µs).

## Install

```bash
pip install transcription-sync          # core (pure Python, zero deps)
pip install transcription-sync[audio]   # + numpy for the amplitude helper
```

## Usage

```python
from transcription_sync import VisemeStream

vs = VisemeStream(fps=60)

# Feed transcription chunks as they stream from your LLM / ASR:
for chunk in llm_transcription_stream:
    vs.feed_text(chunk)
    for frame in vs.frames():            # frame = {ARKit channel: value in [0,1]}
        send_to_avatar(frame)            # drive your VRM / MetaHuman / ARKit rig
```

Track loudness so the mouth stays shut during silence:

```python
from transcription_sync import rms_envelope          # needs [audio] extra

vs.set_amplitude(rms_envelope(pcm_chunk))             # update from your audio stream
```

Real-time pacing (blocking driver; feed text from another thread):

```python
vs.drive_realtime(on_frame=send_to_avatar, stop=lambda: turn_done)
```

Offline / deterministic (for tests and previews):

```python
frames = VisemeStream().render_text("Hello there!", amplitude=0.3)
```

## Output

14 mouth-region ARKit channels are driven (`JawOpen`, `MouthClose`, `MouthFunnel`,
`MouthPucker`, `MouthStretch*`, `MouthUpperUp*`, `MouthLowerDown*`, `MouthShrugUpper`,
`MouthRollLower`, `MouthDimple*`); all other channels are left untouched so an
expression/emotion layer can drive the rest of the face additively.

## How it works

1. **Viseme table** (`visemes.py`) — each grapheme maps to an articulatory pose
   (bilabials close the lips, rounded vowels funnel/pucker, spread vowels stretch…).
   Design constants from phonetics, not fitted to any corpus.
2. **Streaming engine** (`engine.py`) — advances one character per tick at ~13 chars/s
   (≈150 wpm), scales each pose by the live amplitude envelope, and exponentially
   smooths between poses.
3. **Amplitude helper** (`amplitude.py`, optional) — RMS envelope from PCM with a
   fast-attack / slow-release follower.

## Scope & limitations

- Grapheme-driven: ideal for shallow-orthography input or where the LLM streams text;
  pair with a G2P front-end for deep orthographies.
- Geometric/articulatory lip-sync, not a perceptual-quality talking-head model.
- Drives mouth blendshapes only; rendering and the rest of the face are yours.

## Paper and reproducibility

This repository accompanies the paper *TranscriptionSync: Training-Free Lip
Synchronization for Native-Audio Large Language Models and a Layer-Wise Evaluation
Against Video-Measured Articulation* (M. Arslan). Besides the streaming viseme engine
above (the rate-scheduled "RATE" variant of the paper), it contains:

- `paper/tools/` — the evaluation modules: condition generators (amplitude-only,
  RATE, TranscriptionSync Layers 1–3, MFA oracle), Layer-2/3 benchmark, GRID
  processing and regressor training.
- `paper/revision/` — the scripts of the revised evaluation: wall-clock Layer-1
  latency, streaming CTC alignment of the Live API transcription, video-measured
  GRID evaluation (including a Python port of the Audio2Face-3D post-processing),
  SyncNet offset test, re-training on synchronous audio and the re-collected Gemini
  Live evaluation. See `paper/revision/README.md`.

The host assistant, A.T.L.A.S, lives in a separate repository
(https://github.com/wpu-research/A.T.L.A.S; its base code derives from Atlas-MK37 by
FatihMakes and is licensed CC BY-NC 4.0). Everything in this repository is MIT.

## License

MIT © Murat Arslan
