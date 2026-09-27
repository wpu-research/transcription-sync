"""Statistical analysis backing the Section V-C claims (Table 4).

Recomputes, from the per-utterance evaluation outputs:
  - per-condition LVD means with percentile-bootstrap 95% CIs;
  - paired (by utterance, n=50) two-sided Wilcoxon signed-rank tests for LVD and
    |lag| contrasts against TSYNC, Holm-corrected across the LVD family;
  - rank-biserial effect sizes and paired Cohen's dz;
  - LSE-D paired contrasts over utterances scored by both conditions (n=45).

Inputs:  paper/eval_work/metrics.json (lvd/lag/timing, 50 utterances)
         paper/eval_work/results.json (lse_d/lse_c, 300 videos)
Run:     python tools/stats_table4.py
"""
import json
import os
import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
EVAL = os.path.normpath(os.path.join(HERE, "..", "eval_work"))
rng = np.random.default_rng(0)

m = json.load(open(os.path.join(EVAL, "metrics.json")))
r = json.load(open(os.path.join(EVAL, "results.json")))
utts = sorted(m.keys())


def lvd(c):
    return np.array([m[u]["lvd"][c] for u in utts])


def lag(c):
    return np.array([m[u]["lag"][c]["lag_ms"] for u in utts])


def boot_ci(x, B=10000):
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def paired(a, b):
    d = a - b
    w = stats.wilcoxon(a, b, zero_method="wilcox", correction=True)
    nz = d[d != 0]
    rb = 1 - (2 * w.statistic) / (len(nz) * (len(nz) + 1) / 2)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) else 0.0
    return w.pvalue, rb, dz, d.mean(), boot_ci(d)


def holm(pmap):
    items = sorted(pmap.items(), key=lambda kv: kv[1])
    k = len(items)
    return {name: min(1.0, p * (k - i)) for i, (name, p) in enumerate(items)}


print("LVD mean [95% CI]")
for c in ["amp", "rate", "tsync", "tsync-dur", "tsync-coart"]:
    x = lvd(c); lo, hi = boot_ci(x)
    print(f"  {c:12s} {x.mean():.3f} [{lo:.3f}, {hi:.3f}]")

raw = {}
detail = {}
for name, c in {"AMP": "amp", "RATE": "rate", "TSYNC-coart": "tsync-coart",
                "TSYNC-dur": "tsync-dur"}.items():
    p, rb, dz, md, ci = paired(lvd("tsync"), lvd(c))
    raw[name] = p
    detail[name] = (rb, dz, md, ci)
hc = holm(raw)
print("\nLVD paired vs TSYNC (Holm-corrected)")
for name in raw:
    rb, dz, md, ci = detail[name]
    print(f"  TSYNC vs {name:11s} Δ={md:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] "
          f"p={raw[name]:.2e} (Holm {hc[name]:.2e}) r={rb:+.2f} dz={dz:+.2f}")

print("\n|lag| paired vs TSYNC")
for name, c in {"RATE": "rate", "TSYNC-coart": "tsync-coart"}.items():
    p, rb, dz, md, ci = paired(np.abs(lag("tsync")), np.abs(lag(c)))
    print(f"  TSYNC vs {name:11s} Δ|lag|={md:+.1f}ms [{ci[0]:+.1f},{ci[1]:+.1f}] p={p:.2e} r={rb:+.2f}")

print("\nLSE-D paired (common utterances)")
for name, c in {"AMP": "amp", "RATE": "rate", "MFA": "mfa"}.items():
    a = {u: r[f"{u}_tsync"]["lse_d"] for u in utts if r[f"{u}_tsync"]["lse_d"] is not None}
    b = {u: r[f"{u}_{c}"]["lse_d"] for u in utts if r[f"{u}_{c}"]["lse_d"] is not None}
    com = sorted(set(a) & set(b))
    aa = np.array([a[u] for u in com]); bb = np.array([b[u] for u in com])
    print(f"  TSYNC vs {name:4s} n={len(com)} {aa.mean():.2f} vs {bb.mean():.2f} "
          f"p={stats.wilcoxon(aa, bb).pvalue:.2e}")
