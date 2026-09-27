"""B2 extra condition 'learned_sync': causal Transformer re-trained on synchronous
GRID audio + video-measured labels (E4, video/transformer seed 0), CPU inference.
Writes b2/cond_extra/<uid>.npz (key learned_sync) + per-utterance CPU time."""
import json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'atlas/paper/tools'))
import grid_train as gtn
torch.set_num_threads(1)
ck = torch.load(HERE / 'e4/runs/video/transformer_s0/best.pt', map_location='cpu')
net = gtn.CausalTransformer(); net.load_state_dict(ck['model']); net.eval()
out = HERE / 'b2/cond_extra'; out.mkdir(exist_ok=True)
times = {}
for p in sorted((HERE / 'b2/cond').glob('*.npz')):
    uid = p.stem
    n = len(np.load(p, allow_pickle=True)['amp'])
    with tempfile.NamedTemporaryFile(suffix='.wav') as tf:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(HERE / 'b2/wav16' / f'{uid}.wav'), '-ar', '25000', tf.name], check=True)
        x, sr = gtn.read_wav(Path(tf.name))
    mel = torch.from_numpy(gtn.log_mel_60fps(x, sr)).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        P = net(mel)[0].numpy()
    times[uid] = {'ms': (time.perf_counter() - t0) * 1000, 'frames': len(P)}
    P = P[:n] if len(P) >= n else np.vstack([P, np.repeat(P[-1:], n - len(P), 0)])
    np.savez_compressed(out / f'{uid}.npz', learned_sync=P)
json.dump(times, open(out / 'cpu_times.json', 'w'))
ms = np.array([v['ms'] / v['frames'] for v in times.values()])
print('done', len(times), 'utts; CPU (1 thread) per-frame ms median %.3f p95 %.3f' % (np.median(ms), np.percentile(ms, 95)))
