"""Parallel SyncNet scoring for B5 (same commands/regexes as run_eval.stage_syncnet)."""
import json, re, subprocess, sys
from pathlib import Path
SN = Path('/tmp/syncnet_python'); W = Path('/home/wpu/tsync_exp/b5/work')
wi, nw = int(sys.argv[1]), int(sys.argv[2])
vids = sorted((W / 'videos').glob('*.mp4'))[wi::nw]
out = W / f'results_{wi}.json'
res = json.loads(out.read_text()) if out.exists() else {}
dd = f'/tmp/sn_data_{wi}'
for v in vids:
    if v.stem in res: continue
    subprocess.run([sys.executable, 'run_pipeline.py', '--videofile', str(v), '--reference', v.stem, '--data_dir', dd,
                    '--min_track', '40', '--overwrite'], cwd=SN, capture_output=True, text=True)
    r = subprocess.run([sys.executable, 'run_syncnet.py', '--videofile', str(v), '--reference', v.stem, '--data_dir', dd],
                       cwd=SN, capture_output=True, text=True)
    o = r.stdout + r.stderr
    a, b, c = (re.search(p, o) for p in (r'AV offset:\s*(-?\d+)', r'Min dist:\s*([\d.]+)', r'Confidence:\s*([\d.]+)'))
    res[v.stem] = ({'av_offset_frames': int(a.group(1)), 'lse_d': float(b.group(1)), 'lse_c': float(c.group(1))}
                   if a and b and c else {'error': 'no_score'})
    out.write_text(json.dumps(res)); print(v.stem, res[v.stem], flush=True)
