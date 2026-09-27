"""Measure MFA forced-alignment latency on the 50-utterance evaluation corpus.

MFA is the LVD alignment oracle of Section V-C (highest fidelity by construction).
This script quantifies its *latency* for the latency x alignment plane of Table 9:
MFA is non-causal — it needs the complete utterance AND its transcript before
emitting a single boundary — so its added latency = utterance duration (buffer)
+ alignment compute. The compute term is reported batch-amortized (one model load
shared across the corpus), the fair number for a session that aligns many turns.

Prereq:  conda env `mfa` with the `english_us_arpa` acoustic model + dictionary,
         and paper/eval_work/mfa_corpus/spk/{*.wav,*.lab} (already in the repo).
Run:     conda activate mfa && python tools/latency_mfa.py
Reports: alignment wall-clock, per-utterance compute, mean utterance duration
         (buffer), and total added latency. Does not overwrite eval_work/mfa_out.
"""
import glob
import json
import os
import subprocess
import tempfile
import time
import wave

HERE = os.path.dirname(__file__)
EVAL = os.path.normpath(os.path.join(HERE, "..", "eval_work"))
CORPUS = os.path.join(EVAL, "mfa_corpus")
ACOUSTIC = DICT = "english_us_arpa"


def mean_utt_duration_ms():
    wavs = glob.glob(os.path.join(CORPUS, "spk", "*.wav"))
    durs = []
    for w in wavs:
        with wave.open(w, "rb") as f:
            durs.append(f.getnframes() / float(f.getframerate()) * 1e3)
    return sum(durs) / len(durs), len(durs)


def main():
    buffer_ms, n = mean_utt_duration_ms()
    out = tempfile.mkdtemp(prefix="mfa_lat_")
    t0 = time.perf_counter()
    r = subprocess.run(
        ["mfa", "align", "--clean", "--single_speaker", CORPUS, DICT, ACOUSTIC, out],
        capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise SystemExit(f"mfa align failed:\n{r.stderr[-1500:]}")
    compute_per_utt = wall / n * 1e3
    total_added = buffer_ms + compute_per_utt
    res = {
        "n": n,
        "align_wall_s": round(wall, 2),
        "compute_ms_per_utt": round(compute_per_utt, 1),
        "buffer_ms": round(buffer_ms, 1),
        "total_added_ms": round(total_added, 1),
        "causal": False,
        "note": "non-causal: needs whole utterance + transcript; LVD oracle (Table 4). "
                "compute is batch-amortized (one model load across the corpus).",
    }
    print(json.dumps(res, indent=2))
    print(f"\nMFA added latency ≈ {total_added/1000:.1f} s "
          f"(buffer {buffer_ms/1000:.1f} s + compute {compute_per_utt/1000:.2f} s)")


if __name__ == "__main__":
    main()
