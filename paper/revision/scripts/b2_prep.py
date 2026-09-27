"""B2 preparation on the GRID held-out subset (s29-s34).

For every utterance with a downloaded video:
  - extracts the video's own audio track (untrimmed, synchronous with the
    frames) -> b2/wav16/<spk>_<utt>.wav (16 kHz mono int16)
  - writes the transcript from the *correct* alignment folder (Zenodo's
    alignments.zip mislabels 22/34 speaker folders; see d2_overlap.json)
    -> b2/mfa_corpus/spk/<spk>_<utt>.lab
  - extracts MediaPipe ARKit blendshapes (14 mouth channels) at 25 fps,
    resampled to 60 fps -> b2/gt/<spk>_<utt>.npy
Usage: venv/bin/python b2_prep.py <spk>
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'atlas/paper/tools'))
from grid_video_gt import BlendshapeExtractor, ensure_model, resample_60fps  # noqa

G = HERE / 'grid'
B2 = HERE / 'b2'
spk = sys.argv[1]
best = json.load(open(G / 'd2_overlap.json'))            # spk -> [n, same, best_dir, n_best]
align_dir = G / 'alignments' / best[spk][2]
for d in ('wav16', 'gt', 'mfa_corpus/spk'):
    (B2 / d).mkdir(parents=True, exist_ok=True)

model = ensure_model(HERE / 'face_landmarker.task')
ext = BlendshapeExtractor(model)
n = 0
for v in sorted((G / 'video' / spk).glob('*.mpg')):
    u = v.stem
    uid = f'{spk}_{u}'
    al = align_dir / f'{u}.align'
    if not al.exists():
        continue
    words = [l.split()[2] for l in al.read_text().splitlines()
             if len(l.split()) == 3 and l.split()[2] not in ('sil', 'sp')]
    wav = B2 / 'wav16' / f'{uid}.wav'
    if not wav.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(v), '-ac', '1', '-ar', '16000',
                        '-sample_fmt', 's16', str(wav)], check=True)
    (B2 / 'mfa_corpus/spk' / f'{uid}.lab').write_text(' '.join(words))
    lnk = B2 / 'mfa_corpus/spk' / f'{uid}.wav'
    if not lnk.exists():
        os.symlink(wav, lnk)
    gt = B2 / 'gt' / f'{uid}.npy'
    if not gt.exists():
        B25, fps, det = ext.extract('video', v)
        if B25 is None or det < 0.8:
            print('skip (face)', uid, det)
            continue
        np.save(gt, resample_60fps(B25, fps, 0.0))
    n += 1
print(spk, 'done', n)
