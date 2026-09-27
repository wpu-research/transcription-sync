# TranscriptionSync — revision experiments

Scripts behind the revised evaluation of *TranscriptionSync: Training-Free Lip
Synchronization for Native-Audio Large Language Models*. No data or model weights
are included; every script downloads or derives what it needs.

## Main findings of the revision

* **Layer-1 latency.** `paper/tools/layer1_bench.py` omits the 3.2 s window fill time, so the reported 214 ms P95 is an artifact; measured in wall-clock time, sliding-window Whisper finalization has a P95 of ≈ 3.4 s on natural speech (English and Turkish alike).
* **Streaming CTC alignment of the API transcription** (new Layer-1 variant, `scripts/layer1_stream.py`) runs on a CPU at ≈ 45 ms per 160 ms hop: P95 latency 0.48 s on 50 re-collected Gemini Live responses (89% of word starts within 100 ms of MFA; 92% on 25 Turkish responses).
* **Video-measured ground truth (GRID, 593 held-out utterances, MediaPipe ARKit).** TSYNC-CTC 0.2642 beats AMP 0.2673 and RATE 0.2684 (LVD_bs); Audio2Face-3D 0.2628 and the MFA oracle 0.2621 remain slightly better. With Whisper as Layer 1, TSYNC (0.2716) is behind the baselines.
* **SyncNet.** LSE-C/LSE-D do not respond to ±200 ms offsets on the untextured avatar; the offset estimate does (slope 1.00, +58 ms bias).
* **GRID distribution.** `audio_25k` is silence-trimmed (mean 432 ms) relative to `.align`/video; 22/34 folders of `alignments.zip` hold another speaker's alignments. The V-G regressors were re-trained on the videos' own audio.

The revised manuscript is kept in the A.T.L.A.S repository (`paper/revision/manuscript/`). `paper/tools/` mirrors the evaluation modules of A.T.L.A.S (`paper/tools/`, commit afb5e60), written by the same author.

## Layout expected at run time

```
<root>/
  atlas/paper/tools -> <this repo>/paper/tools   (the scripts import these modules;
                    e.g. `mkdir -p atlas/paper && ln -s <this repo>/paper/tools atlas/paper/tools`)
  atlas/public/human.glb   evaluation avatar (Git LFS object of https://github.com/wpu-research/A.T.L.A.S)
  grid/             GRID subsets fetched by grid/fetch_*.py (Zenodo record 3625687)
  fleurs/           FLEURS en_us / tr_tr test subsets
  scripts/*, live/*, a2f/*   (this folder)
  venv/             Python 3.12: torch, torchaudio, faster-whisper, mediapipe 0.10.x
                    (protobuf 4.25), onnxruntime(-gpu), scipy, soundfile, pygltflib,
                    cmudict, remotezip, google-genai, matplotlib
  MFA               conda env "mfa" with english_us_arpa and turkish_mfa models
```

## Experiments

| Paper section | What | Scripts |
|---|---|---|
| V-C (SyncNet) | Controlled offsets of the AMP trajectory, SyncNet scoring | `b5.py`, `b5_syncnet_par.py` |
| V-D | All conditions vs. MediaPipe ARKit ground truth on GRID s29–s34 (593 utt.) | `b2_prep.py`, `b2_conditions.py`, `b2_extra.py`, `b2_a2f.py`, `b2_learned_sync.py`, `b2_ctc.py`, `b2_eval.py` |
| V-E | GRID edition check (trimmed audio vs. .align, mislabeled folders) | see `grid/fetch_subset.py` and the notes in the paper |
| V-F | Wall-clock Layer-1 latency (sliding window, LocalAgreement, streaming CTC alignment) | `layer1_wall.py`, `layer1_stream.py`, `c2_run.sh`, `tr_mfa_prep.py`, `tr_ctc_score.py` |
| V-G | Audio-to-blendshape regression re-trained on synchronous audio (3 seeds) | `e4_build.py`, `e4_train.py`, `e4_figs.py`; full-data run `e4full_*` |
| V-C (live) | Re-collected Gemini Live material (EN/TR), content-matched timing | `live/collect_audio_live.py`, `live/live_pipeline.py` |

`a2f/a2f_py.py` is a Python port of the NVIDIA Audio2Face-3D SDK post-processing and
blendshape solve (MIT) that runs the official v2.3 Mark ONNX network (NVIDIA Open
Model License; download it from Hugging Face). It omits blinks, tongue and emotion
inference; see its docstring.

## Gemini access

`live/collect_audio_live.py` reads the API key from `GEMINI_API_KEY` or `~/.gemini_key`.
Never commit a key.

## Known pitfalls reproduced here

* `paper/tools/layer1_bench.py` starts its clock when the first 3.2 s window is
  submitted, so latencies are underestimated; `layer1_wall.py --paper-clock` reproduces
  it, `layer1_wall.py` measures wall-clock latency.
* The Zenodo `audio_25k` edition of GRID is silence-trimmed (mean 432 ms) relative to
  the `.align` files and videos, and 22 of 34 folders in `alignments.zip` belong to
  another speaker. Use the videos' own audio track, and match utterance ids across folders.
