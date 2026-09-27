"""E4: rebuild the Section V-G training data with audio that is synchronous
with the labels (the video's own audio track), for both label sources.

The original pipeline paired labels on the untrimmed video/.align clock with
the silence-trimmed audio_25k wavs (offset mean 432 ms, SD 146 ms; see D2).
Here audio, video-measured labels and rule-derived labels share one clock.

Per utterance (train s1-s28 subset, val s29-s34 subset):
  e4/corpus/<spk>/<utt>.wav   25 kHz mono int16 from the .mpg
  video GT:   MediaPipe 25 fps -> 60 fps (grid_video_gt defaults, no smoothing)
  rule GT:    grid_process.words_to_frames(words from the CORRECT .align, 40 ms)
Writes e4/video_gt.jsonl and e4/rule_gt.jsonl in grid_train.py's format.
Usage: venv/bin/python e4_build.py <spk>   (then: e4_build.py merge)
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'atlas/paper/tools'))
G, E4 = HERE / 'grid', HERE / 'e4_full'
CH = ["jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthStretchLeft", "mouthStretchRight",
      "mouthUpperUpLeft", "mouthUpperUpRight", "mouthLowerDownLeft", "mouthLowerDownRight",
      "mouthShrugUpper", "mouthRollLower", "mouthDimpleLeft", "mouthDimpleRight"]


def frames(A):
    return [{"t": round(i * 1000.0 / 60, 2), "a": [round(float(x), 3) for x in A[i]]} for i in range(len(A))]


if sys.argv[1] == 'merge':
    for src in ('video', 'rule'):
        with open(E4 / f'{src}_gt.jsonl', 'w') as f:
            for p in sorted((E4 / 'parts').glob(f'*_{src}.jsonl')):
                f.write(p.read_text())
        print(src, sum(1 for _ in open(E4 / f'{src}_gt.jsonl')), 'records')
    sys.exit()

from grid_video_gt import BlendshapeExtractor, resample_60fps   # noqa
from grid_process import words_to_frames                         # noqa

spk = sys.argv[1]
num = int(spk[1:])
best = json.load(open(G / 'd2_overlap.json'))
align_dir = G / 'alignments' / best[spk][2]
split = 'val' if num >= 29 else 'train'
(E4 / 'parts').mkdir(parents=True, exist_ok=True)
(E4 / 'corpus' / spk).mkdir(parents=True, exist_ok=True)
b2gt = HERE / 'b2' / 'gt'
ext = BlendshapeExtractor(HERE / 'face_landmarker.task')
fv = open(E4 / 'parts' / f'{spk}_video.jsonl', 'w')
fr = open(E4 / 'parts' / f'{spk}_rule.jsonl', 'w')
n = 0
for v in sorted((G / 'video' / spk).glob('*.mpg')):
    u = v.stem
    al = align_dir / f'{u}.align'
    if not al.exists():
        continue
    wav = E4 / 'corpus' / spk / f'{u}.wav'
    if not wav.exists():
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(v), '-ac', '1', '-ar', '25000',
                        '-sample_fmt', 's16', str(wav)], check=True)
    if (b2gt / f'{spk}_{u}.npy').exists():
        A = np.load(b2gt / f'{spk}_{u}.npy')
    else:
        B25, fps, det = ext.extract('video', v)
        if B25 is None or det < 0.8:
            continue
        A = resample_60fps(B25, fps, 0.0)
    words = [(l.split()[2], int(l.split()[0]) / 25000, int(l.split()[1]) / 25000)
             for l in al.read_text().splitlines() if len(l.split()) == 3 and l.split()[2] not in ('sil', 'sp')]
    base = {"id": f"grid_{spk}_{u}", "lang": "en", "spk": spk, "wav": f"{spk}/{u}.wav",
            "fps": 60, "channels": CH, "split": split}
    fv.write(json.dumps({**base, "src": "video_gt", "frames": frames(A)}) + "\n")
    fr.write(json.dumps({**base, "src": "rule", "frames": words_to_frames(words, 40.0)}) + "\n")
    n += 1
print(spk, split, n)
