"""Data-driven validity check for SyncNet (LSE-D/LSE-C) on the untextured VRM
avatar, backing the Section V-C analysis. Uses only the conditions already
rendered/scored — no re-running of SyncNet required.

Three tests:
  1. Can SyncNet separate articulatorily-EMPTY motion (AMP, whose JawOpen *is*
     the audio envelope) from the MFA forced-alignment oracle? (paired Wilcoxon
     on LSE-C and LSE-D; expect INDISTINGUISHABLE -> metric fails on quality.)
  2. Is SyncNet's offset estimate biased on construction-perfect sync? AMP has
     true offset 0 by construction; test estimated offset vs 0.
  3. Does the offset estimate track the independently measured mouth-audio lag?
     (Pearson/Spearman of SyncNet offset vs metrics.json lag.)

Inputs: paper/eval_work/{results.json, metrics.json}
Run:    python tools/syncnet_validity.py
"""
import json
import os
import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
EVAL = os.path.normpath(os.path.join(HERE, "..", "eval_work"))
MSPF = 1000.0 / 25.0  # ms per frame at SyncNet's 25 fps input
rng = np.random.default_rng(0)

r = json.load(open(os.path.join(EVAL, "results.json")))
m = json.load(open(os.path.join(EVAL, "metrics.json")))
utts = sorted(m.keys())


def field(metric, c):
    return {u: r[f"{u}_{c}"][metric] for u in utts if r[f"{u}_{c}"].get(metric) is not None}


def boot(d, B=10000):
    return np.percentile([rng.choice(d, len(d), replace=True).mean() for _ in range(B)], [2.5, 97.5])


def paired(a, b):
    com = sorted(set(a) & set(b))
    aa = np.array([a[u] for u in com]); bb = np.array([b[u] for u in com])
    return len(com), aa.mean(), bb.mean(), stats.wilcoxon(aa, bb).pvalue, boot(aa - bb)


print("TEST 1 — AMP (empty) vs MFA (oracle): can SyncNet tell them apart?")
for met in ["lse_c", "lse_d"]:
    n, a, b, p, ci = paired(field(met, "amp"), field(met, "mfa"))
    print(f"  {met.upper()}: AMP={a:.2f} MFA={b:.2f} Δ95%CI[{ci[0]:+.2f},{ci[1]:+.2f}] p={p:.3f} "
          f"-> {'INDISTINGUISHABLE' if p > 0.05 else 'separable'}")
for c in ["rate", "tsync"]:
    n, a, b, p, ci = paired(field("lse_c", "amp"), field("lse_c", c))
    print(f"  LSE-C AMP={a:.2f} > {c.upper()}={b:.2f}? p={p:.1e}")

print("\nTEST 2 — offset bias on construction-perfect sync (true offset = 0)")
for c in ["amp", "mfa"]:
    off = np.array(list(field("av_offset_frames", c).values()), float) * MSPF
    print(f"  {c.upper()}: est. offset {off.mean():+.1f} ms 95%CI{tuple(round(x,1) for x in boot(off))} "
          f"vs 0 p={stats.wilcoxon(off).pvalue:.1e}")

print("\nTEST 3 — SyncNet offset vs independently measured mouth-audio lag")
X, Y = [], []
for u in utts:
    for c in ["amp", "rate", "tsync", "tsync-dur", "tsync-coart", "mfa"]:
        o = r[f"{u}_{c}"].get("av_offset_frames")
        lag = m[u]["lag"].get(c, {}).get("lag_ms")
        if o is not None and lag is not None:
            X.append(o * MSPF); Y.append(lag)
X, Y = np.array(X), np.array(Y)
print(f"  n={len(X)}  Pearson r={stats.pearsonr(X, Y).statistic:+.3f}  "
      f"Spearman ρ={stats.spearmanr(X, Y).statistic:+.3f}")
