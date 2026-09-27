import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = Path('/home/wpu/tsync_exp'); sys.path.insert(0, str(HERE / 'atlas/paper/tools'))
import grid_eval as ge
R = json.load(open(HERE / 'e4/e4_results.json'))
FIG = Path('/tmp/claude-1000/-home-wpu/e74b4496-86cd-4f22-834b-ca2f1745e2ee/scratchpad/rev/fig')
# Fig A: LVD mean and velocity ratio, mean ± SD over 3 seeds
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
groups = [('rule', 'transformer'), ('rule', 'lstm'), ('video', 'transformer'), ('video', 'lstm')]
lab = ['Rule\nTransformer', 'Rule\nBiLSTM', 'Video\nTransformer', 'Video\nBiLSTM']
col = ['#4a90d9', '#b0855b', '#1f5fa8', '#8d5b2a']
for a, (key, name) in zip(ax, [('lvd_mean', 'LVD μ (lower is better)'), ('lag_mean', 'Mean lag (ms)'), ('vel_ratio', 'Velocity ratio (1.0 = reference)')]):
    m = [np.mean([R[f'{g}/{h}/{s}'][key] for s in (0, 1, 2)]) for g, h in groups]
    sd = [np.std([R[f'{g}/{h}/{s}'][key] for s in (0, 1, 2)], ddof=1) for g, h in groups]
    a.bar(range(4), m, yerr=sd, color=col, capsize=4); a.set_xticks(range(4)); a.set_xticklabels(lab, fontsize=8); a.set_title(name, fontsize=10)
    if key == 'vel_ratio': a.axhline(1.0, ls='--', color='k', lw=1)
    if key == 'lag_mean': a.axhline(0, color='k', lw=0.8)
fig.suptitle('Audio-to-blendshape regression re-trained on synchronous audio (mean ± SD over 3 seeds; n = 593 held-out utterances)', fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / 'fig_e4_summary.png', dpi=200)
# Fig B: lag histograms, seed 0
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for a, src in zip(ax, ('rule', 'video')):
    for arch, c, nm in (('transformer', '#1f5fa8', 'Transformer (causal)'), ('lstm', '#d9822b', 'BiLSTM (non-causal)')):
        r = ge.evaluate_run(HERE / f'e4/runs/{src}/{arch}_s0', HERE / f'e4/cache_{src}', 'cpu', 10 ** 9, 0)
        a.hist(r['lags'], bins=21, range=(-250, 250), alpha=0.55, color=c, label=f"{nm}: {r['lag_mean']:+.0f} ± {r['lag_std']:.0f} ms (n={len(r['lags'])})")
    a.axvline(0, color='k', lw=0.8); a.set_xlabel('Lag (ms), positive = prediction late'); a.set_title(f'{"Rule-derived" if src == "rule" else "Video-measured"} labels, seed 0', fontsize=10); a.legend(fontsize=7.5)
ax[0].set_ylabel('Number of utterances')
fig.tight_layout(); fig.savefig(FIG / 'fig_e4_lag.png', dpi=200)
print('ok')
