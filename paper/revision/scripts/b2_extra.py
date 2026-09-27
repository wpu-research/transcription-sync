"""B2 diagnostics: two extra TSYNC variants added to b2/cond/*.npz
  tsync_small_trim  Layer-1 input trimmed at the energy onset with the offset
                    restored (grid_layer_validation.trim_leading_silence) -
                    the onset handling Section IV-B describes but
                    make_conditions.py does not implement
  tsync_refwords    MFA word boundaries -> Layer 2 (Eq. 4) -> Layer 3: an
                    oracle Layer 1 that isolates Layers 2-3
Usage: venv/bin/python b2_extra.py <worker> <n_workers>
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE / 'atlas/paper/tools'
sys.path.insert(0, str(T)); sys.path.insert(0, str(T / 'render'))
import make_conditions as mc                                             # noqa
from grid_layer_validation import trim_leading_silence, parse_textgrid_tier  # noqa
from faster_whisper import WhisperModel                                   # noqa
from faster_whisper.audio import decode_audio                             # noqa

B2 = HERE / 'b2'
wi, nw = int(sys.argv[1]), int(sys.argv[2])
CH = list(mc.CHANNELS)
arr = lambda fr, n: np.array([[f.get(c, 0.0) for c in CH] for f in mc.pad_to(fr, n)], dtype=np.float32)
model = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=2)
for p in sorted((B2 / 'cond').glob('*.npz'))[wi::nw]:
    z = dict(np.load(p, allow_pickle=True))
    if 'tsync_refwords' in z:
        continue
    uid = p.stem
    audio = decode_audio(str(B2 / 'wav16' / f'{uid}.wav'), sampling_rate=mc.SR)
    n = mc.n_frames_for(audio)
    trimmed, off = trim_leading_silence(audio)
    tr = [(w, a + off, b + off) for w, a, b in mc.streaming_word_triples(trimmed, 'en', model)]
    z['tsync_small_trim'] = arr(mc.kernel_frames(mc.timed_phonemes_from_triples(tr, 'en'), 'en', n), n)
    z['words_small_trim'] = np.array(tr, dtype=object)
    ref = [(w.lower(), a, b) for w, a, b in parse_textgrid_tier(B2 / 'mfa_out/spk' / f'{uid}.TextGrid', 'words') if w.strip()]
    z['tsync_refwords'] = arr(mc.kernel_frames(mc.timed_phonemes_from_triples(ref, 'en'), 'en', n), n)
    np.savez_compressed(p, **z)
    print(uid, 'ok', flush=True)
