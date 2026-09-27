"""B2: generate every condition's 14-channel ARKit trajectory for the GRID
held-out subset, using the paper's own generators (make_conditions.py) on the
video's synchronous audio track.

Conditions: amp, rate, tsync_tiny, tsync_small, tsync-dur_small,
tsync-coart_small, mfa, learned (released causal Transformer checkpoint,
paper/train_results/grid_runs/best.pt; trained on GRID s1-s28).
Output: b2/cond/<uid>.npz with one (T,14) array per condition.
Usage: venv/bin/python b2_conditions.py <worker_index> <n_workers>
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE / 'atlas/paper/tools'
sys.path.insert(0, str(T)); sys.path.insert(0, str(T / 'render'))
import make_conditions as mc                      # noqa
import grid_train as gtn                          # noqa
import torch                                      # noqa
from faster_whisper import WhisperModel           # noqa
from faster_whisper.audio import decode_audio     # noqa

B2 = HERE / 'b2'
wi, nw = int(sys.argv[1]), int(sys.argv[2])
out_dir = B2 / 'cond'; out_dir.mkdir(exist_ok=True)
CH = list(mc.CHANNELS)


def arr(frames, n):
    frames = mc.pad_to(frames, n)
    return np.array([[f.get(c, 0.0) for c in CH] for f in frames], dtype=np.float32)


torch.set_num_threads(2)
models = {m: WhisperModel(m, device='cpu', compute_type='int8', cpu_threads=2) for m in ('tiny', 'small')}
ck = torch.load(T.parent / 'train_results/grid_runs/best.pt', map_location='cpu')
net = gtn.CausalTransformer(); net.load_state_dict(ck['model']); net.eval()

uids = sorted(p.stem for p in (B2 / 'gt').glob('*.npy'))[wi::nw]
for uid in uids:
    out = out_dir / f'{uid}.npz'
    tg = B2 / 'mfa_out/spk' / f'{uid}.TextGrid'
    if out.exists() or not tg.exists():
        continue
    wav = B2 / 'wav16' / f'{uid}.wav'
    audio = decode_audio(str(wav), sampling_rate=mc.SR)
    n = mc.n_frames_for(audio)
    transcript = (B2 / 'mfa_corpus/spk' / f'{uid}.lab').read_text()
    R = {'amp': arr(mc.amp_schedule(audio), n),
         'rate': arr(mc.rate_schedule(audio, transcript), n)}
    words = {}
    for m, model in models.items():
        tr = mc.streaming_word_triples(audio, 'en', model)
        words[m] = tr
        R[f'tsync_{m}'] = arr(mc.kernel_frames(mc.timed_phonemes_from_triples(tr, 'en'), 'en', n), n)
    tr = words['small']
    R['tsync-dur_small'] = arr(mc.kernel_frames(mc.timed_phonemes_from_triples(tr, 'en', uniform=True), 'en', n), n)
    R['tsync-coart_small'] = arr(mc.step_frames(mc.timed_phonemes_from_triples(tr, 'en'), n), n)
    R['mfa'] = arr(mc.kernel_frames(mc.mfa_timed_phonemes(tg, 'en'), 'en', n), n)
    # learned baseline expects the corpus sample rate (25 kHz)
    with tempfile.NamedTemporaryFile(suffix='.wav') as tf:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(wav), '-ar', '25000', tf.name], check=True)
        x, sr = gtn.read_wav(Path(tf.name))
    with torch.no_grad():
        P = net(torch.from_numpy(gtn.log_mel_60fps(x, sr)).unsqueeze(0))[0].numpy()
    R['learned'] = P[:n] if len(P) >= n else np.vstack([P, np.repeat(P[-1:], n - len(P), 0)])
    np.savez_compressed(out, **R, words_tiny=np.array(words['tiny'], dtype=object),
                        words_small=np.array(words['small'], dtype=object))
    print(uid, 'ok', flush=True)
print('worker', wi, 'done')
