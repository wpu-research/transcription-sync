"""End-to-end added-latency benchmark for the two lip-sync engines (Section V-C).

Measures *added pipeline latency* — how long after audio is available each engine
takes to start driving the avatar's mouth. This is DISTINCT from alignment error
(the 1380 ms live word-timing MAE / LVD geometric fidelity reported elsewhere):

    pipeline latency = when does the mouth start moving, relative to audio
    alignment  error = once moving, how well do mouth and audio line up

The two engines have structurally different latency profiles:

  SYS1 (Live Viseme / TranscriptionSync)
      buffer    = 0      — streams per transcription chunk, no turn-level wait
      inference = ~0     — deterministic phoneme→ARKit table lookup (microseconds)
      transport = WS hop — one WebSocket broadcast to the browser
    → added latency is just transport + one frame quantization (tens of ms)

  SYS2 (HuBERT neural blendshapes)
      buffer    = whole utterance must complete before inference can run (seconds)
      inference = 326M-param Transformer forward pass (server `generation_ms`)
      transport = HTTP round-trip to the sidecar + WS hop to the browser
    → added latency = buffer + inference + transport

Everything here is MEASURED on the same 50 evaluation utterances used by the rest
of the paper (paper/eval_audio/*.wav, 24 kHz — the live Gemini output rate), except
the two structural zeros (SYS1 buffer/inference), which are quantified separately as
a micro-benchmark of the rule-table lookup to show they are negligible.

Prereq:  lipsync server running —
    cd lipsync/audio2lipsync/python/src && \
        python server.py --ckpt ../checkpoints/best.pt --stats-dir ../stats
Run:     python tools/latency_eval.py
Output:  printed table + paper/eval_work/latency.json
"""
import base64
import glob
import json
import os
import time
import wave

import numpy as np
import requests

HERE = os.path.dirname(__file__)
AUDIO_DIR = os.path.normpath(os.path.join(HERE, "..", "eval_audio"))
EVAL_DIR = os.path.normpath(os.path.join(HERE, "..", "eval_work"))
LIPSYNC_URL = "http://localhost:8765"
FPS = 60
GAIN = 1.2

rng = np.random.default_rng(0)


def boot_ci(x, B=10000):
    """Percentile bootstrap 95% CI of the mean — same estimator as stats_table4.py."""
    x = np.asarray(x, dtype=float)
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def summary(name, x, unit="ms"):
    x = np.asarray(x, dtype=float)
    lo, hi = boot_ci(x)
    print(f"  {name:28s} {x.mean():9.2f} [{lo:8.2f}, {hi:8.2f}] {unit}"
          f"   (median {np.median(x):.2f}, max {x.max():.2f})")
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)],
            "median": float(np.median(x)), "max": float(x.max()),
            "n": int(len(x))}


def wav_duration_s(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def wav_bytes_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── SYS1 structural-zero micro-benchmark ──────────────────────────────────────
# SYS1's per-frame work is a dict lookup in the phoneme→ARKit viseme table plus
# building the broadcast payload. Time it over many iterations to show it is
# negligible vs SYS2's buffer+inference (which are seconds).
def sys1_lookup_micro(n=200_000):
    VISEME = {  # representative subset of main.py's _VISEME map
        'a': {'JawOpen': 0.60, 'MouthLowerDownLeft': 0.35, 'MouthLowerDownRight': 0.35},
        'm': {'JawOpen': 0.00, 'MouthClose': 0.92, 'MouthShrugUpper': 0.15},
        'o': {'JawOpen': 0.42, 'MouthFunnel': 0.38, 'MouthPucker': 0.18},
        '_': {'JawOpen': 0.12, 'MouthShrugUpper': 0.08},
    }
    chars = "amerhaba_o" * 100
    t0 = time.perf_counter()
    for _ in range(n // len(chars)):
        for ch in chars:
            target = dict(VISEME.get(ch, VISEME['_']))
            _ = {k: min(1.0, v * 1.1) for k, v in target.items()}
    dt = time.perf_counter() - t0
    return dt / n * 1e3  # ms per frame


def main():
    try:
        h = requests.get(f"{LIPSYNC_URL}/health", timeout=3).json()
    except Exception as e:
        raise SystemExit(f"lipsync server not reachable at {LIPSYNC_URL}: {e}\n"
                         f"Start it first (see module docstring).")
    print(f"lipsync server: {h}\n")

    wavs = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    if not wavs:
        raise SystemExit(f"no eval audio in {AUDIO_DIR}")
    print(f"Benchmarking {len(wavs)} utterances "
          f"(eval_audio/, {FPS}fps, gain {GAIN})\n")

    buffer_ms, infer_ms, http_ms, transport_ms = [], [], [], []
    per_utt = {}

    for i, path in enumerate(wavs):
        utt = os.path.splitext(os.path.basename(path))[0]
        dur_s = wav_duration_s(path)
        payload = {"audio_base64": wav_bytes_b64(path), "fps": FPS, "gain": GAIN}

        t0 = time.perf_counter()
        r = requests.post(f"{LIPSYNC_URL}/generate", json=payload, timeout=120)
        rtt = (time.perf_counter() - t0) * 1e3  # ms, full HTTP round-trip
        r.raise_for_status()
        d = r.json()
        gen = float(d.get("generation_ms", 0.0))

        # SYS2 components for this utterance
        b = dur_s * 1e3                  # buffer: whole turn must complete
        transport = max(0.0, rtt - gen)  # round-trip minus server inference
        buffer_ms.append(b)
        infer_ms.append(gen)
        http_ms.append(rtt)
        transport_ms.append(transport)
        per_utt[utt] = {"duration_s": round(dur_s, 3), "buffer_ms": round(b, 2),
                        "inference_ms": round(gen, 2), "http_rtt_ms": round(rtt, 2),
                        "transport_ms": round(transport, 2), "n_frames": d.get("n_frames")}
        print(f"  [{i+1:2d}/{len(wavs)}] {utt:12s} dur={dur_s:5.2f}s "
              f"infer={gen:7.1f}ms rtt={rtt:7.1f}ms", flush=True)

    sys2_added = np.array(buffer_ms) + np.array(http_ms)  # buffer + (inf+transport)
    sys1_lookup_ms = sys1_lookup_micro()

    print("\n" + "=" * 78)
    print("SYS2 (HuBERT) — added pipeline latency, N={}".format(len(wavs)))
    print("=" * 78)
    res = {"n": len(wavs), "sys2": {}, "sys1": {}}
    res["sys2"]["buffer_ms"]    = summary("buffer (turn duration)", buffer_ms)
    res["sys2"]["inference_ms"] = summary("inference (generation_ms)", infer_ms)
    res["sys2"]["transport_ms"] = summary("transport (rtt - inference)", transport_ms)
    res["sys2"]["http_rtt_ms"]  = summary("HTTP round-trip total", http_ms)
    res["sys2"]["total_added_ms"] = summary("TOTAL added (buffer+rtt)", sys2_added)

    print("\n" + "=" * 78)
    print("SYS1 (Live Viseme / TranscriptionSync) — added pipeline latency")
    print("=" * 78)
    print(f"  buffer (turn wait)           {'0.00':>9}                          ms"
          f"   (structural: streams per chunk)")
    print(f"  inference (table lookup)     {sys1_lookup_ms*1e3:9.2f}                          us"
          f"   (structural: deterministic)")
    print(f"  → SYS1 adds no turn-level buffering and no neural inference;")
    print(f"    its added latency is dominated by the shared WebSocket hop")
    print(f"    (same transport SYS2 also pays) plus one {1000/FPS:.1f} ms frame quantum.")
    res["sys1"]["buffer_ms"] = 0.0
    res["sys1"]["inference_us_per_frame"] = round(sys1_lookup_ms * 1e3, 4)
    res["sys1"]["frame_quantum_ms"] = round(1000.0 / FPS, 2)

    print("\n" + "=" * 78)
    print("HONESTY NOTE: this benchmark measures pipeline LATENCY only — when the")
    print("mouth starts moving relative to audio. ALIGNMENT error once moving")
    print("(live 1380 ms word-timing MAE; geometric LVD fidelity) is a separate")
    print("axis, reported in stats_table4.py / Section V-C. Do not conflate them.")
    print("=" * 78)

    os.makedirs(EVAL_DIR, exist_ok=True)
    out = os.path.join(EVAL_DIR, "latency.json")
    with open(out, "w") as f:
        json.dump({"summary": res, "per_utterance": per_utt}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
