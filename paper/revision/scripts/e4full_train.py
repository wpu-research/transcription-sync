"""E4: retrain the Section V-G regressors on synchronous audio, 3 seeds x
2 architectures x 2 label sources, and evaluate with the paper's own
grid_eval.evaluate_run (Table 8 columns). Writes e4/e4_results.json.
Usage: venv/bin/python e4_train.py
"""
import json, random, sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'atlas/paper/tools'))
import grid_train as gt   # noqa
import grid_eval as ge    # noqa

E4 = HERE / 'e4_full'
res_path = E4 / 'e4_results.json'
results = json.loads(res_path.read_text()) if res_path.exists() else {}
for src in ('rule', 'video'):
    cache = E4 / f'cache_{src}'
    if not (cache / 'train.npz').exists():
        gt.prepare(E4 / f'{src}_gt.jsonl', E4 / 'corpus', cache)
    for arch in ('transformer', 'lstm'):
        for seed in (0, 1, 2):
            key = f'{src}/{arch}/{seed}'
            if key in results:
                continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            run = E4 / 'runs' / src / f'{arch}_s{seed}'
            gt.train(Namespace(data=str(E4 / f'{src}_gt.jsonl'), corpus=str(E4 / 'corpus'), cache=str(cache),
                               arch=arch, epochs=30, batch=32, lr=3e-4, patience=6, workers=0,
                               device='cuda', runs=str(run)))
            r = ge.evaluate_run(run, cache, 'cuda', 10 ** 9, 0)
            results[key] = {k: float(r[k]) for k in ('val_mse', 'val_mae', 'lvd_mean', 'lvd_p95',
                                                     'lag_mean', 'lag_std', 'vel_ratio')}
            results[key]['n_val'] = int(r['n_utts'])
            res_path.write_text(json.dumps(results, indent=1))
            print('RESULT', key, results[key], flush=True)

print('\nTABLE 8 (re-run, synchronous audio): mean ± SD over 3 seeds')
cols = ('val_mse', 'val_mae', 'lvd_mean', 'lvd_p95', 'lag_mean', 'lag_std', 'vel_ratio')
for src in ('rule', 'video'):
    for arch in ('transformer', 'lstm'):
        rs = [results[f'{src}/{arch}/{s}'] for s in (0, 1, 2)]
        print(f'{src:5s} {arch:11s} ' + '  '.join(
            f"{c}={np.mean([r[c] for r in rs]):.4f}±{np.std([r[c] for r in rs], ddof=1):.4f}" for c in cols))
