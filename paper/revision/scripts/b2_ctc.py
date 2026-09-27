"""B2 condition 'tsync_ctc': Layer 1 = streaming CTC alignment (wav2vec2-base, CPU, hop 160 ms,
guard 100 ms, commit after next word starts) of the known transcript; Layers 2-3 unchanged.
Writes b2/cond_extra/<uid>.npz key tsync_ctc (merged with existing keys) + latency list."""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
T = HERE / 'atlas/paper/tools'
sys.path.insert(0, str(T)); sys.path.insert(0, str(T / 'render')); sys.path.insert(0, str(HERE))
import make_conditions as mc
from layer1_stream import Aligner, run_ctc
from faster_whisper.audio import decode_audio
torch.set_num_threads(4)
al = Aligner('w2v', 'cpu')
CH = list(mc.CHANNELS)
lat, comp, trip = [], [], {}
for p in sorted((HERE / 'b2/cond').glob('*.npz')):
    uid = p.stem
    n = len(np.load(p, allow_pickle=True)['amp'])
    au = decode_audio(str(HERE / 'b2/wav16' / f'{uid}.wav'), sampling_rate=16000)
    words = (HERE / 'b2/mfa_corpus/spk' / f'{uid}.lab').read_text().split()
    out, l, c = run_ctc(al, au, words, 0.16, 0.10, sil=0.20)
    lat += l; comp += c; trip[uid] = out
    fr = mc.kernel_frames(mc.timed_phonemes_from_triples([(w.lower(), s, e) for w, s, e in out], 'en'), 'en', n)
    A = np.array([[f.get(ch, 0.0) for ch in CH] for f in mc.pad_to(fr, n)], dtype=np.float32)
    ex = HERE / 'b2/cond_extra' / f'{uid}.npz'
    z = dict(np.load(ex)) if ex.exists() else {}
    z['tsync_ctc_sil'] = A; np.savez_compressed(ex, **z)
L = np.array(lat) * 1000; C = np.array(comp) * 1000
json.dump({k: [(w, float(s), float(e)) for w, s, e in v] for k, v in trip.items()}, open(HERE / 'b2/ctc_sil_words.json', 'w'))
print('GRID tsync_ctc_sil: n_words %d  L median %.0f P95 %.0f max %.0f ms | C median %.0f P95 %.0f ms' % (len(L), np.median(L), np.percentile(L, 95), L.max(), np.median(C), np.percentile(C, 95)))
