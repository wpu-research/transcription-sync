"""B2: add NVIDIA Audio2Face-3D v2.3 (Mark, regression; Python port a2f/a2f_py.py)
as condition 'a2f' to b2/cond/<uid>.npz. 30 fps ARKit -> 60 fps (linear),
14 mouth channels in make_conditions.CHANNELS order. Usage: b2_a2f.py <wi> <nw>"""
import json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'a2f')); sys.path.insert(0, str(HERE / 'atlas/paper/tools/render')); sys.path.insert(0, str(HERE / 'atlas/paper/tools'))
from a2f_py import A2F, FPS
from make_conditions import CHANNELS
wi, nw = int(sys.argv[1]), int(sys.argv[2])
m = A2F(threads=2)
cols = [m.pose_names.index(c[0].lower() + c[1:]) for c in CHANNELS]
log = []
for p in sorted((HERE / 'b2/cond').glob('*.npz'))[wi::nw]:
    z = dict(np.load(p, allow_pickle=True))
    if 'a2f' in z:
        continue
    a, sr = sf.read(HERE / 'b2/wav16' / f'{p.stem}.wav', dtype='float32')
    t0 = time.perf_counter(); W = m.run(a); dt = time.perf_counter() - t0
    n = len(z['amp'])
    t30 = np.arange(len(W)) / FPS; t60 = np.arange(n) / 60.0
    A = np.stack([np.interp(t60, t30, W[:, c]) for c in cols], 1).astype(np.float32)
    z['a2f'] = A
    np.savez_compressed(p, **z)
    log.append({'uid': p.stem, 'sec': dt, 'frames30': len(W), 'audio_s': len(a) / sr, 'device': 'cpu'})
    print(p.stem, f'{dt:.2f}s', flush=True)
json.dump(log, open(HERE / f'b2/a2f_timing_{wi}.json', 'w'))
print('done', wi)
