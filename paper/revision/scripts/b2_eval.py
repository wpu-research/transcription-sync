"""B2 evaluation: every condition vs video-measured ARKit articulation (GRID
s29-s34 subset) + D4 signed timing on the same material.

Metrics per utterance and condition
  r_<ch>     Pearson correlation with the MediaPipe trajectory (scale-free)
  lvd_bs     mean per-frame L2 distance in 14-ch blendshape space after a
             per-channel affine calibration pred -> GT, fitted
             leave-one-speaker-out (never on the scored speaker)
  lvd_mesh   same, as lip-vertex displacement on the evaluation avatar
             (human.glb, top-400 lip vertices; compute_metrics.load_lip_deltas)
  lag_ms     cross-correlation lag of JawOpen vs GT jawOpen (+ = prediction late)
Writes b2/b2_results.json and prints the summary tables.
"""
import difflib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
T = HERE / 'atlas/paper/tools'
sys.path.insert(0, str(T)); sys.path.insert(0, str(T / 'render'))
from compute_metrics import load_lip_deltas                    # noqa
from grid_layer_validation import parse_textgrid_tier          # noqa
from make_conditions import CHANNELS                           # noqa

B2 = HERE / 'b2'
CONDS = ['amp', 'rate', 'learned', 'learned_sync', 'a2f', 'tsync_ctc', 'tsync_ctc_sil', 'tsync_tiny', 'tsync_small', 'tsync_small_trim', 'tsync-dur_small',
         'tsync-coart_small', 'tsync_refwords', 'mfa']
REF = 'tsync_small'
FPS = 60
rng = np.random.default_rng(0)

data = {}
for p in sorted((B2 / 'cond').glob('*.npz')):
    uid = p.stem
    z = dict(np.load(p, allow_pickle=True))
    z.update(dict(np.load(B2 / 'cond_extra' / f'{uid}.npz')))
    gt = np.load(B2 / 'gt' / f'{uid}.npy')
    n = min(len(gt), *(len(z[c]) for c in CONDS))
    data[uid] = {'gt': gt[:n], **{c: z[c][:n] for c in CONDS},
                 'words_tiny': list(z['words_tiny']), 'words_small': list(z['words_small']),
                 'words_small_trim': list(z['words_small_trim'])}
uids = sorted(data)
spk_of = {u: u.split('_')[0] for u in uids}
print('utterances:', len(uids), 'speakers:', sorted(set(spk_of.values())))

# ── leave-one-speaker-out affine calibration per condition & channel ───────
def calib(cond):
    out = {}
    for s in sorted(set(spk_of.values())):
        tr = [u for u in uids if spk_of[u] != s]
        X = np.vstack([data[u][cond] for u in tr]); Y = np.vstack([data[u]['gt'] for u in tr])
        ab = []
        for k in range(14):
            x = X[:, k]
            if x.std() < 1e-6:
                ab.append((0.0, Y[:, k].mean()))
            else:
                a, b = np.polyfit(x, Y[:, k], 1)
                ab.append((max(a, 0.0), b))          # no sign flips
        out[s] = np.array(ab)
    return out

lip, _ = load_lip_deltas(HERE / 'atlas/public/human.glb')
D = np.stack([lip.get(ch, np.zeros_like(next(iter(lip.values())))) for ch in CHANNELS])  # (14,K,3)

def mesh(Bs):
    return np.einsum('tc,ckd->tkd', Bs, D)

def xlag(a, b, maxlag=30):
    a = a - a.mean(); b = b - b.mean()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    best, bl = -np.inf, 0
    for l in range(-maxlag, maxlag + 1):
        x, y = (a[l:], b[:len(b) - l]) if l >= 0 else (a[:l], b[-l:])
        c = np.corrcoef(x, y)[0, 1]
        if c > best:
            best, bl = c, l
    return bl / FPS * 1000

M = defaultdict(dict)
for c in CONDS:
    cal = calib(c)
    for u in uids:
        P, G = data[u][c], data[u]['gt']
        ab = cal[spk_of[u]]
        Pc = np.clip(P * ab[:, 0] + ab[:, 1], 0, 1)
        r = {}
        for k, ch in enumerate(CHANNELS):
            r[ch] = float(np.corrcoef(P[:, k], G[:, k])[0, 1]) if P[:, k].std() > 1e-6 and G[:, k].std() > 1e-6 else np.nan
        M[u][c] = {'r': r,
                   'lvd_bs': float(np.linalg.norm(Pc - G, axis=1).mean()),
                   'lvd_mesh': float(np.linalg.norm(mesh(Pc) - mesh(G), axis=2).mean()),
                   'lag_ms': xlag(P[:, 0], G[:, 0])}

def arr(c, key, ch=None):
    v = [M[u][c]['r'][ch] if key == 'r' else M[u][c][key] for u in uids]
    return np.array(v, dtype=float)

def boot(x, B=10000):
    x = x[~np.isnan(x)]
    return np.percentile([rng.choice(x, len(x)).mean() for _ in range(B)], [2.5, 97.5])

rows = []
print('\nTABLE B2  (mean [95% CI]; n utterances)')
print(f"{'cond':18s} {'LVD_bs':>18s} {'LVD_mesh':>12s} {'r_jaw':>7s} {'r_close':>8s} {'r_mean14':>9s} {'lag ms (med)':>13s}")
for c in CONDS:
    lb, lm = arr(c, 'lvd_bs'), arr(c, 'lvd_mesh')
    rj, rc = arr(c, 'r', 'JawOpen'), arr(c, 'r', 'MouthClose')
    rmean = np.array([np.nanmean([M[u][c]['r'][ch] for ch in CHANNELS]) for u in uids])
    lag = arr(c, 'lag_ms')
    ci = boot(lb)
    rows.append({'cond': c, 'lvd_bs': lb.mean(), 'lvd_bs_ci': ci.tolist(), 'lvd_mesh': lm.mean(),
                 'r_jaw': np.nanmean(rj), 'r_close': np.nanmean(rc), 'r_mean14': np.nanmean(rmean),
                 'lag_med': float(np.nanmedian(lag)), 'lag_mean': float(np.nanmean(lag)), 'lag_sd': float(np.nanstd(lag))})
    print(f"{c:18s} {lb.mean():.4f} [{ci[0]:.4f},{ci[1]:.4f}] {lm.mean():12.5f} {np.nanmean(rj):7.3f} {np.nanmean(rc):8.3f} {np.nanmean(rmean):9.3f} {np.nanmedian(lag):7.0f} ({np.nanmean(lag):.0f}±{np.nanstd(lag):.0f})")

print(f'\nPaired Wilcoxon vs {REF} on LVD_bs (Holm) and r_mean14')
raw, det = {}, {}
ref = arr(REF, 'lvd_bs')
refr = np.array([np.nanmean([M[u][REF]['r'][ch] for ch in CHANNELS]) for u in uids])
for c in CONDS:
    if c == REF:
        continue
    x = arr(c, 'lvd_bs'); d = ref - x
    w = stats.wilcoxon(ref, x)
    nz = d[d != 0]; rb = 1 - 2 * w.statistic / (len(nz) * (len(nz) + 1) / 2)
    xr = np.array([np.nanmean([M[u][c]['r'][ch] for ch in CHANNELS]) for u in uids])
    raw[c] = w.pvalue; det[c] = (d.mean(), boot(d), rb, (refr - xr).mean(), stats.wilcoxon(refr, xr).pvalue)
order = sorted(raw, key=raw.get); holm = {c: min(1, raw[c] * (len(order) - i)) for i, c in enumerate(order)}
for c in raw:
    dm, ci, rb, dr, pr = det[c]
    print(f'  {REF} - {c:18s} ΔLVD={dm:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] p_holm={holm[c]:.1e} r_rb={rb:+.2f} | Δr_mean14={dr:+.3f} p={pr:.1e}')

# ── D4 / D3: signed word-start timing vs MFA, content-matched ──────────────
print('\nD4 signed word-start error vs MFA (pred - ref; negative = visemes early), content-matched')
timing = {}
for m in ('tiny', 'small', 'small_trim'):
    signed, per_utt_med, resid = [], [], []
    match = []
    for u in uids:
        ref_w = [(w.lower(), a, b) for w, a, b in parse_textgrid_tier(B2 / 'mfa_out/spk' / f'{u}.TextGrid', 'words') if w.strip()]
        hyp = [(str(w).lower(), float(a), float(b)) for w, a, b in data[u][f'words_{m}']]
        sm = difflib.SequenceMatcher(a=[w for w, _, _ in ref_w], b=[w for w, _, _ in hyp], autojunk=False)
        e = []
        for blk in sm.get_matching_blocks():
            for i in range(blk.size):
                e.append((hyp[blk.b + i][1] - ref_w[blk.a + i][1]) * 1000)
        match.append(len(e) / max(1, len(ref_w)))
        if e:
            signed += e; per_utt_med.append(np.median(e)); resid += list(np.array(e) - np.median(e))
    s = np.array(signed); off = np.median(s)
    timing[m] = {'n_words': len(s), 'mae': float(np.abs(s).mean()), 'median_signed': float(off),
                 'iqr': np.percentile(s, [25, 75]).tolist(), 'frac_early': float((s < 0).mean()),
                 'mae_after_global_comp': float(np.abs(s - off).mean()),
                 'within_utt_residual_mae': float(np.abs(resid).mean()),
                 'frac_within_bt1359': float(((s >= -125) & (s <= 45)).mean()),
                 'frac_within_bt1359_after_comp': float((((s - off) >= -125) & ((s - off) <= 45)).mean()),
                 'match_rate': float(np.mean(match))}
    t = timing[m]
    print(f"  {m:5s} n={t['n_words']} MAE={t['mae']:.0f} ms | median signed={t['median_signed']:+.0f} IQR=[{t['iqr'][0]:+.0f},{t['iqr'][1]:+.0f}] "
          f"early={t['frac_early']*100:.0f}% | MAE after global comp={t['mae_after_global_comp']:.0f} | within-utt resid MAE={t['within_utt_residual_mae']:.0f} "
          f"| in BT.1359 window: {t['frac_within_bt1359']*100:.0f}% -> {t['frac_within_bt1359_after_comp']*100:.0f}% after comp | match={t['match_rate']*100:.0f}%")

json.dump({'n_utt': len(uids), 'table': rows, 'holm': holm, 'timing': timing,
           'per_utt': M}, open(B2 / 'b2_results.json', 'w'), indent=1, default=float)
print('\nwrote', B2 / 'b2_results.json')
