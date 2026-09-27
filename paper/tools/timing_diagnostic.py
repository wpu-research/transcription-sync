"""Diagnostic for the live word-timing error (Section V-C).

Shows the 1380.9 ms word-start MAE against MFA is a per-utterance, near-constant
offset (word-start MAE == word-end MAE, r ~ 1.00), broadly distributed (not
outlier-driven), and larger for longer turn types — consistent with an absolute
speech-onset anchoring offset rather than within-utterance drift.

Input: paper/eval_work/metrics.json
Run:   python tools/timing_diagnostic.py
"""
import json
import os
import collections
import numpy as np

HERE = os.path.dirname(__file__)
m = json.load(open(os.path.normpath(os.path.join(HERE, "..", "eval_work", "metrics.json"))))
utts = sorted(m.keys())

st = np.array([m[u]["timing"]["tsync"]["mae_start_ms"] for u in utts])
en = np.array([m[u]["timing"]["tsync"]["mae_end_ms"] for u in utts])

print(f"word-start MAE  mean={st.mean():.1f}  SD={st.std():.1f}  median={np.median(st):.1f}")
print(f"word-end   MAE  mean={en.mean():.1f}")
print(f"start vs end correlation r = {np.corrcoef(st, en)[0, 1]:.4f}  "
      f"(~1.00 => per-utterance constant shift, not drift)")

order = np.sort(st)[::-1]
print(f"5 worst utterances contribute {order[:5].sum() / st.sum() * 100:.1f}% of summed error "
      f"(not outlier-driven)")

b = collections.defaultdict(list)
for u in utts:
    b[''.join(c for c in u if not c.isdigit())].append(m[u]["timing"]["tsync"]["mae_start_ms"])
print("by batch (mean ms):", {k: round(np.mean(v)) for k, v in sorted(b.items())})
