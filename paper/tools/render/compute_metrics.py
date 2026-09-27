"""
Section V-C metrics that need no rendering: LVD (lip vertex distance), word
timing error, and audio-visual lag. Consumes the blendshape schedules from
make_conditions.py (work/schedules/*.json), the MFA TextGrids (work/mfa_out),
and the source audio (eval_audio/*.wav). Writes work/metrics.json.

  LVD       per-frame L2 displacement of a lip-region vertex set (human.glb,
            head_lod0 mesh) under condition `cond` vs the MFA-oracle condition
            for the same utterance, averaged over frames and vertices. Lower
            is better (closer to the oracle mouth shape trajectory).
  timing    for tsync / tsync-dur / tsync-coart (which carry the causal Layer-1
            'words' field): mean absolute error of word start/end times vs the
            MFA 'words' tier, paired by order (same convention as Section V-D).
  lag       cross-correlation peak between the JawOpen trajectory and the
            audio RMS envelope, in ms (all conditions, all utterances).

Usage:
  python compute_metrics.py --audio paper/eval_audio --work paper/eval_work \
      --model public/human.glb
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # paper/tools

from make_conditions import CHANNELS, rms_envelope, n_frames_for, FPS, SR
from grid_layer_validation import parse_textgrid_tier

CONDITIONS = ['amp', 'rate', 'tsync', 'tsync-dur', 'tsync-coart', 'mfa']
TIMED_CONDITIONS = ['tsync', 'tsync-dur', 'tsync-coart']


# ── lip vertex set from human.glb ───────────────────────────────────────────
def load_lip_deltas(glb_path: Path, top_k: int = 400):
    """Returns (channel -> (K,3) morph delta array) for the K vertices most
    affected (by summed displacement magnitude across the 14 channels) on the
    head_lod0 mesh, plus the chosen vertex indices."""
    import pygltflib
    g = pygltflib.GLTF2().load(str(glb_path))
    mesh = g.meshes[0]  # head_lod0: the morph-bearing face mesh
    prim = mesh.primitives[0]
    target_names = mesh.extras['targetNames']
    blob = g.get_data_from_buffer_uri(g.buffers[0].uri) if g.buffers[0].uri else g.binary_blob()

    def read_accessor(idx):
        acc = g.accessors[idx]
        bv = g.bufferViews[acc.bufferView]
        offset = bv.byteOffset + (acc.byteOffset or 0)
        count = acc.count
        arr = np.frombuffer(blob, dtype=np.float32, count=count * 3,
                             offset=offset).reshape(count, 3)
        return arr

    name_to_idx = {n: i for i, n in enumerate(target_names)}
    deltas = {}
    for ch in CHANNELS:
        lc = ch[:1].lower() + ch[1:]
        if lc not in name_to_idx:
            continue
        ti = name_to_idx[lc]
        deltas[ch] = read_accessor(prim.targets[ti]['POSITION'])

    n_verts = next(iter(deltas.values())).shape[0]
    total = np.zeros(n_verts)
    for arr in deltas.values():
        total += np.linalg.norm(arr, axis=1)
    lip_idx = np.argsort(total)[::-1][:top_k]
    lip_idx = lip_idx[total[lip_idx] > 0]
    return {ch: arr[lip_idx] for ch, arr in deltas.items()}, lip_idx


def lip_trajectory(frames, lip_deltas):
    """frames: list of {channel: value} dicts -> (n_frames, K, 3) displacement."""
    K = next(iter(lip_deltas.values())).shape[0]
    traj = np.zeros((len(frames), K, 3), dtype=np.float32)
    for ch, deltas in lip_deltas.items():
        vals = np.array([f.get(ch, 0.0) for f in frames], dtype=np.float32)
        traj += vals[:, None, None] * deltas[None, :, :]
    return traj


def resample_frames(frames, src_fps, n_out, out_fps=FPS):
    if src_fps == out_fps and len(frames) == n_out:
        return frames
    out = []
    for f in range(n_out):
        t = f / out_fps
        src = min(len(frames) - 1, round(t * src_fps))
        out.append(frames[src])
    return out


# ── LVD ──────────────────────────────────────────────────────────────────────
def compute_lvd(sched_dir: Path, uid: str, lip_deltas):
    mfa_path = sched_dir / f'{uid}_mfa.json'
    if not mfa_path.exists():
        return {}
    mfa = json.loads(mfa_path.read_text())
    mfa_traj = lip_trajectory(mfa['frames'], lip_deltas)
    n_total = len(mfa['frames'])

    out = {}
    for cond in CONDITIONS:
        if cond == 'mfa':
            continue
        p = sched_dir / f'{uid}_{cond}.json'
        if not p.exists():
            continue
        sched = json.loads(p.read_text())
        frames = resample_frames(sched['frames'], sched['fps'], n_total)
        traj = lip_trajectory(frames, lip_deltas)
        dist = np.linalg.norm(traj - mfa_traj, axis=2)  # (n_frames, K)
        out[cond] = float(dist.mean())
    return out


# ── timing error vs MFA word tier ───────────────────────────────────────────
def compute_timing(sched_dir: Path, mfa_textgrid: Path | None, uid: str):
    if mfa_textgrid is None or not mfa_textgrid.exists():
        return {}
    ref_words = [(t, a, b) for t, a, b in parse_textgrid_tier(mfa_textgrid, 'words') if t.strip()]
    out = {}
    for cond in TIMED_CONDITIONS:
        p = sched_dir / f'{uid}_{cond}.json'
        if not p.exists():
            continue
        sched = json.loads(p.read_text())
        words = sched.get('words', [])
        n = min(len(words), len(ref_words))
        if n == 0:
            continue
        start_err = [abs(words[i][1] - ref_words[i][1]) for i in range(n)]
        end_err = [abs(words[i][2] - ref_words[i][2]) for i in range(n)]
        out[cond] = {'mae_start_ms': float(np.mean(start_err) * 1000),
                      'mae_end_ms': float(np.mean(end_err) * 1000),
                      'n_words': n}
    return out


# ── lag: JawOpen vs RMS envelope cross-correlation ──────────────────────────
def compute_lag(sched_dir: Path, audio, uid: str, max_lag_frames: int = 30):
    env = rms_envelope(audio)
    n_total = n_frames_for(audio)
    env = env - env.mean()

    out = {}
    for cond in CONDITIONS:
        p = sched_dir / f'{uid}_{cond}.json'
        if not p.exists():
            continue
        sched = json.loads(p.read_text())
        frames = resample_frames(sched['frames'], sched['fps'], n_total)
        jaw = np.array([f.get('JawOpen', 0.0) for f in frames], dtype=np.float64)
        jaw = jaw - jaw.mean()
        if jaw.std() == 0 or env.std() == 0:
            continue
        best_lag, best_corr = 0, -np.inf
        for lag in range(-max_lag_frames, max_lag_frames + 1):
            if lag >= 0:
                a, b = jaw[lag:], env[:len(env) - lag]
            else:
                a, b = jaw[:len(jaw) + lag], env[-lag:]
            if len(a) < 10:
                continue
            corr = np.corrcoef(a, b)[0, 1]
            if not np.isnan(corr) and corr > best_corr:
                best_corr, best_lag = corr, lag
        out[cond] = {'lag_ms': float(best_lag / FPS * 1000), 'corr': float(best_corr)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--model', default=str(REPO_ROOT / 'public' / 'human.glb'))
    args = ap.parse_args()

    from faster_whisper.audio import decode_audio

    audio_dir, work = Path(args.audio), Path(args.work)
    sched_dir = work / 'schedules'
    manifest = json.loads((audio_dir / 'manifest.json').read_text())

    print('[metrics] loading lip vertex set from', args.model)
    lip_deltas, lip_idx = load_lip_deltas(Path(args.model))
    print(f'[metrics] {len(lip_idx)} lip vertices, channels: {list(lip_deltas.keys())}')

    results = {}
    for uid in sorted(manifest.keys()):
        if not any((sched_dir / f'{uid}_{c}.json').exists() for c in CONDITIONS):
            continue
        audio = decode_audio(str(audio_dir / f'{uid}.wav'), sampling_rate=SR)
        tg = work / 'mfa_out' / 'spk' / f'{uid}.TextGrid'
        results[uid] = {
            'lvd': compute_lvd(sched_dir, uid, lip_deltas),
            'timing': compute_timing(sched_dir, tg if tg.exists() else None, uid),
            'lag': compute_lag(sched_dir, audio, uid),
        }
        print(f'[metrics] {uid}: lvd={results[uid]["lvd"]} '
              f'lag={ {c: v["lag_ms"] for c, v in results[uid]["lag"].items()} }')

    out_path = work / 'metrics.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f'[metrics] {len(results)} utterances -> {out_path}')

    # quick aggregate summary
    for cond in CONDITIONS:
        lvds = [r['lvd'][cond] for r in results.values() if cond in r['lvd']]
        lags = [r['lag'][cond]['lag_ms'] for r in results.values() if cond in r['lag']]
        if lvds or lags:
            lvd_s = f'LVD={np.mean(lvds):.5f}' if lvds else 'LVD=n/a'
            lag_s = f'lag={np.mean(lags):.1f}ms' if lags else 'lag=n/a'
            print(f'  {cond:12s} {lvd_s}  {lag_s}  (n={len(lvds)}/{len(lags)})')
    for cond in TIMED_CONDITIONS:
        starts = [r['timing'][cond]['mae_start_ms'] for r in results.values() if cond in r['timing']]
        if starts:
            print(f'  {cond:12s} word-start MAE = {np.mean(starts):.1f} ms (n={len(starts)})')


if __name__ == '__main__':
    main()
