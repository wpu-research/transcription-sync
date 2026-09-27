"""B5: controlled SyncNet sensitivity test on the evaluation avatar.

20 FLEURS-en utterances (natural read speech, independent of the paper's
material). Conditions rendered with the paper's own renderer and scored with
the paper's own SyncNet stage (run_eval.py):
  amp_shift{+-ms}  AMP trajectory displaced by a known offset
                   (positive = visemes late), 9 offsets in [-200, +200] ms
  rate, tsync (Whisper small, as in the paper's live evaluation), mfa  at 0 ms
Stages: schedules | render | syncnet | report
Usage: venv/bin/python b5.py <stage> [workers]
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE / 'atlas/paper/tools'
sys.path.insert(0, str(T)); sys.path.insert(0, str(T / 'render'))
B5 = HERE / 'b5'
AUDIO = B5 / 'audio'
SHIFTS = [-200, -120, -80, -40, 0, 40, 80, 120, 200]
man = json.load(open(HERE / 'fleurs/en_us/manifest.json'))
UIDS = sorted(man)[:20]
stage = sys.argv[1]


def shifted(frames, ms, rest):
    k = round(ms / 1000 * 60)
    if k > 0:
        return [dict(rest)] * k + frames[:len(frames) - k]
    if k < 0:
        return frames[-k:] + [dict(rest)] * (-k)
    return frames


if stage == 'schedules':
    import make_conditions as mc
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    AUDIO.mkdir(parents=True, exist_ok=True)
    sd = B5 / 'work/schedules'; sd.mkdir(parents=True, exist_ok=True)
    model = WhisperModel('small', device='cpu', compute_type='int8')
    manifest = {}
    for u in UIDS:
        wav = AUDIO / f'{u}.wav'
        if not wav.exists():
            wav.symlink_to(HERE / f'fleurs/en_us/{u}.wav')
        audio = decode_audio(str(wav), sampling_rate=mc.SR)
        n = mc.n_frames_for(audio)
        tr = (B5 / 'mfa_corpus/spk' / f'{u}.lab').read_text()
        manifest[u] = {'transcript': tr}
        amp = mc.pad_to(mc.amp_schedule(audio), n)
        rest = dict(amp[0]); rest.update({k: 0.0 for k in rest})
        out = {f'amp_shift{s:+d}': shifted(amp, s, rest) for s in SHIFTS}
        out['rate'] = mc.pad_to(mc.rate_schedule(audio, tr), n)
        trip = mc.streaming_word_triples(audio, 'en', model)
        out['tsync'] = mc.kernel_frames(mc.timed_phonemes_from_triples(trip, 'en'), 'en', n)
        out['mfa'] = mc.kernel_frames(mc.mfa_timed_phonemes(B5 / 'mfa_out/spk' / f'{u}.TextGrid', 'en'), 'en', n)
        for c, fr in out.items():
            (sd / f'{u}_{c}.json').write_text(json.dumps({'fps': 60, 'condition': c, 'frames': mc.pad_to(fr, n)}))
        print(u, 'ok', flush=True)
    (AUDIO / 'manifest.json').write_text(json.dumps(manifest))

elif stage == 'render':
    import multiprocessing as mp
    import run_eval as re_
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    (B5 / 'work/videos').mkdir(parents=True, exist_ok=True)
    jobs = [tuple(p.stem.split('_', 1)) for p in sorted((B5 / 'work/schedules').glob('*.json'))]
    jobs = [(u, c) for u, c in jobs if not (B5 / 'work/videos' / f'{u}_{c}.mp4').exists()]
    print(len(jobs), 'videos to render', flush=True)
    chunks = [jobs[i::workers] for i in range(workers)]
    ps = [mp.Process(target=re_._render_worker, args=(ch, str(AUDIO), str(B5 / 'work'), i)) for i, ch in enumerate(chunks)]
    [p.start() for p in ps]; [p.join() for p in ps]

elif stage == 'syncnet':
    import run_eval as re_
    re_.stage_syncnet(AUDIO, B5 / 'work')

elif stage == 'report':
    from scipy import stats
    r = json.loads((B5 / 'work/results.json').read_text())
    def get(c, key):
        return {u: r[f'{u}_{c}'][key] for u in UIDS if f'{u}_{c}' in r and key in r[f'{u}_{c}']}
    print('condition        n  LSE-D  LSE-C  est.offset(ms)')
    rows = {}
    for c in [f'amp_shift{s:+d}' for s in SHIFTS] + ['rate', 'tsync', 'mfa']:
        d, cf, off = get(c, 'lse_d'), get(c, 'lse_c'), get(c, 'av_offset_frames')
        if not d:
            continue
        o = np.array(list(off.values())) * 40.0
        rows[c] = {'n': len(d), 'lse_d': float(np.mean(list(d.values()))), 'lse_c': float(np.mean(list(cf.values()))),
                   'offset_ms': float(o.mean()), 'offset_sd': float(o.std())}
        print(f"{c:15s} {len(d):3d} {rows[c]['lse_d']:6.2f} {rows[c]['lse_c']:6.2f} {o.mean():+7.0f} ± {o.std():.0f}")
    xs, ys = [], []
    for s in SHIFTS:
        for u, o in get(f'amp_shift{s:+d}', 'av_offset_frames').items():
            xs.append(s); ys.append(o * 40.0)
    xs, ys = np.array(xs), np.array(ys)
    fit = np.polyfit(xs, ys, 1)
    print(f'\nimposed shift vs SyncNet offset: slope {fit[0]:.2f} (ideal 1 or -1 depending on sign convention), '
          f'intercept {fit[1]:+.0f} ms, Pearson r {stats.pearsonr(xs, ys).statistic:+.3f}, n={len(xs)}')
    a0, m0 = get('amp_shift+0', 'lse_c'), get('mfa', 'lse_c')
    com = sorted(set(a0) & set(m0))
    p = stats.wilcoxon([a0[u] for u in com], [m0[u] for u in com]).pvalue if len(com) > 5 else float('nan')
    print(f'AMP(0) vs MFA LSE-C: {np.mean([a0[u] for u in com]):.2f} vs {np.mean([m0[u] for u in com]):.2f}, p={p:.3f}, n={len(com)}')
    lc = {s: get(f'amp_shift{s:+d}', 'lse_c') for s in SHIFTS}
    json.dump({'rows': rows, 'fit': fit.tolist(), 'r': float(stats.pearsonr(xs, ys).statistic),
               'amp0_vs_mfa_p': p}, open(B5 / 'b5_results.json', 'w'), indent=1)
