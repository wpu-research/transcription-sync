"""Aggregate live per-turn latency records into a clean summary table.

Reads the JSONL written by the running app (main.py instrumentation) and reports
mean + percentile-bootstrap 95% CI per metric, split by which engine was active.
Confirms the offline buffer estimate (latency_eval.py) against real Gemini
streaming timing — the live buffer = first-audio-chunk → turn-complete, which
depends on how fast Gemini streams, not on audio playback duration.

Metrics per turn (ms):
  SYS2 (when sys2 active):
    buffer_ms          first audio chunk → turn_complete (live Gemini streaming)
    inference_xport_ms t_dispatch → t_ready (HuBERT inference + transport)
    sys2_added_ms      first audio → mouth ready  (glass-to-glass)
    generation_ms      server-reported pure inference
  SYS1 (when sys1 active):
    text_lead_ms       first audio → first transcription text (+ve: text lags audio)
    sys1_added_ms      first text → first mouth render (queue + ws serialize)

Run:  python tools/latency_live.py [path/to/live_latency.jsonl]
Default path: ../../eval_work/live_latency.jsonl  (BASE_DIR/eval_work)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
DEFAULT = os.path.normpath(os.path.join(HERE, "..", "..", "eval_work", "live_latency.jsonl"))
rng = np.random.default_rng(0)


def boot_ci(x, B=10000):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"),) * 2
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(B)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def col(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def show(name, vals, unit="ms"):
    if not vals:
        print(f"  {name:24s}  (no data)")
        return
    x = np.asarray(vals, dtype=float)
    lo, hi = boot_ci(x)
    print(f"  {name:24s} {x.mean():9.2f} [{lo:8.2f},{hi:8.2f}] {unit}"
          f"  (median {np.median(x):.1f}, n={len(x)})")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        raise SystemExit(f"no live latency log at {path}\n"
                         f"Run the app and drive some turns first "
                         f"(SYS1 and/or SYS2 enabled).")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    sys2 = [r for r in rows if r.get("sys2")]
    sys1 = [r for r in rows if r.get("sys1") and not r.get("sys2")]

    print(f"Live latency log: {path}")
    print(f"Total turns: {len(rows)}  (sys2={len(sys2)}, sys1-only={len(sys1)})\n")

    print("=" * 70)
    print(f"SYS2 (HuBERT) live added latency  —  N={len(sys2)}")
    print("=" * 70)
    show("buffer (audio→turn_end)", col(sys2, "buffer_ms"))
    show("inference+transport", col(sys2, "inference_xport_ms"))
    show("generation_ms (server)", col(sys2, "generation_ms"))
    show("TOTAL added (glass-glass)", col(sys2, "sys2_added_ms"))
    show("audio duration", col(sys2, "audio_dur_ms"))

    print("\n" + "=" * 70)
    print(f"SYS1 (Live Viseme) live added latency  —  N={len(sys1)}")
    print("=" * 70)
    show("text lead (audio→text)", col(sys1, "text_lead_ms"))
    show("SYS1 added (text→render)", col(sys1, "sys1_added_ms"))
    show("audio duration", col(sys1, "audio_dur_ms"))

    print("\n" + "=" * 70)
    print("NOTE: latency only (when mouth starts) — NOT alignment error.")
    print("=" * 70)


if __name__ == "__main__":
    main()
