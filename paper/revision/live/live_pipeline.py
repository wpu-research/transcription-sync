"""Live Gemini re-evaluation (replaces the unreleased paper/eval_audio material).

Stages (python live_pipeline.py <stage> <lang> [worker n_workers]):
  prep        16 kHz wavs, MFA corpus (.lab from the Live API output transcription),
              per-word text-availability times from the recorded event log
  conditions  amp, rate, tsync_small (paper config), tsync_asr (tiny + onset trim),
              tsync_ctc_live (streaming CTC, text used only once it has arrived),
              a2f (en only; Audio2Face-3D port), mfa (oracle)  -> cond/<uid>.npz
  eval        LVD to the MFA-driven mesh, envelope lag, content-matched word timing vs MFA
              (start/end MAE, signed, global-offset compensation), Layer-1 wall-clock
              latency, text lead, utterance statistics -> results_<lang>.json
Directory: live/eval_<lang>/ (collect_audio_live.py output)
"""
import difflib, json, re, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T = ROOT / 'atlas/paper/tools'
sys.path.insert(0, str(T)); sys.path.insert(0, str(T / 'render')); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'a2f'))

stage, lang = sys.argv[1], sys.argv[2]
D = HERE / f'eval_{lang}'
W16 = D / 'wav16'; MC = D / 'mfa_corpus/spk'; MO = D / 'mfa_out/spk'; CD = D / 'cond'
SR_OUT = 24000


def norm_word(w):
    return re.sub(r"[^\w']", '', w.lower())


def words_and_avail(uid):
    """Split the output transcription into words; each word is 'known' at the wall time
    (relative to the first audio chunk) of the text chunk that completes it."""
    ev = json.load(open(D / f'{uid}.events.json'))
    t_audio0 = next(t for t, kind, *_ in ev if kind == 'audio')
    text, avail_chars = '', []
    for t, kind, *rest in ev:
        if kind == 'text':
            chunk = rest[0]
            text += chunk
            avail_chars += [t - t_audio0] * len(chunk)
    words, avail = [], []
    for m in re.finditer(r'\S+', text):
        if norm_word(m.group()):
            words.append(m.group()); avail.append(max(0.0, avail_chars[m.end() - 1]))
    return words, avail, text


IPA_TR = {'a': 'a', 'e': 'e', 'i': 'i', 'ɯ': 'ı', 'ɨ': 'ı', 'o': 'o', 'ø': 'ö', 'œ': 'ö', 'u': 'u', 'y': 'ü',
          'b': 'b', 'p': 'p', 'd': 'd', 't': 't', 'ɡ': 'g', 'g': 'g', 'k': 'k', 'c': 'k', 'ɟ': 'g',
          'f': 'f', 'v': 'v', 'ʋ': 'v', 'w': 'v', 's': 's', 'z': 'z', 'ʃ': 'ş', 'ʒ': 'j', 'h': 'h', 'x': 'h', 'ç': 'h',
          'tʃ': 'ç', 't̠ʃ': 'ç', 'dʒ': 'c', 'd̠ʒ': 'c', 'm': 'm', 'n': 'n', 'ŋ': 'n', 'ɲ': 'n', 'l': 'l', 'ɫ': 'l', 'ʎ': 'l',
          'ɾ': 'r', 'r': 'r', 'ɹ': 'r', 'j': 'y', 'ɰ': 'ğ', 'ɣ': 'ğ', 'ʔ': None,
          'ʏ': 'ü', 'ɛ': 'e', 'ɔ': 'o', 'ɪ': 'i', 'ʊ': 'u', 'ɑ': 'a', 'æ': 'e', 'ɐ': 'a', 'ə': 'e'}


def ipa_key(p):
    import unicodedata
    p = p.strip().replace('ː', '').replace('ʲ', '')
    return ''.join(ch for ch in p if unicodedata.category(ch) != 'Mn')


def mfa_timed(tg, lang, mc):
    if lang == 'en':
        return mc.mfa_timed_phonemes(tg, lang)
    from grid_layer_validation import parse_textgrid_phones
    table, bilset, _ = mc.lang_tables(lang)
    out = []
    for p, a, b in parse_textgrid_phones(tg):
        ph = IPA_TR.get(ipa_key(p), None)
        if ph and ph in table:
            cls, V = table[ph]; out.append((ph, (a + b) / 2, V, ph in bilset))
    return sorted(out, key=lambda x: x[1])


if stage == 'prep':
    for d in (W16, MC):
        d.mkdir(parents=True, exist_ok=True)
    man = json.load(open(D / 'manifest.json'))
    for uid in sorted(man):
        w16 = W16 / f'{uid}.wav'
        if not w16.exists():
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(D / f'{uid}.wav'), '-ac', '1', '-ar', '16000',
                            '-sample_fmt', 's16', str(w16)], check=True)
        words, avail, text = words_and_avail(uid)
        (MC / f'{uid}.lab').write_text(' '.join(norm_word(w) for w in words))
        lnk = MC / f'{uid}.wav'
        if not lnk.exists():
            lnk.symlink_to(w16)
        json.dump({'words': words, 'avail': avail}, open(W16 / f'{uid}.words.json', 'w'), ensure_ascii=False)
    print('prep done', len(man))

elif stage == 'conditions':
    import torch
    import make_conditions as mc
    from grid_layer_validation import trim_leading_silence
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from layer1_stream import Aligner, run_ctc
    wi, nw = int(sys.argv[3]), int(sys.argv[4])
    torch.set_num_threads(2)
    CD.mkdir(exist_ok=True)
    CH = list(mc.CHANNELS)
    arr = lambda fr, n: np.array([[f.get(c, 0.0) for c in CH] for f in mc.pad_to(fr, n)], dtype=np.float32)
    small = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=2)
    tiny = WhisperModel('tiny', device='cpu', compute_type='int8', cpu_threads=2)
    al = Aligner('w2v' if lang == 'en' else 'mms', 'cpu')
    from a2f_py import A2F, FPS as A2F_FPS
    if True:
        a2f = A2F(threads=2); a2f_cols = [a2f.pose_names.index(c[0].lower() + c[1:]) for c in CH]
    for w16 in sorted(W16.glob('*.wav'))[wi::nw]:
        uid = w16.stem; out = CD / f'{uid}.npz'; tg = MO / f'{uid}.TextGrid'
        if out.exists() or not tg.exists():
            continue
        audio = decode_audio(str(w16), sampling_rate=16000); n = mc.n_frames_for(audio)
        wj = json.load(open(W16 / f'{uid}.words.json'))
        transcript = ' '.join(wj['words'])
        R, meta = {}, {}
        R['amp'] = arr(mc.amp_schedule(audio), n)
        R['rate'] = arr(mc.rate_schedule(audio, transcript), n)
        tr = mc.streaming_word_triples(audio, lang, small)
        R['tsync_small'] = arr(mc.kernel_frames(mc.timed_phonemes_from_triples(tr, lang), lang, n), n); meta['words_small'] = tr
        trimmed, off = trim_leading_silence(audio)
        tr = [(w, a + off, b + off) for w, a, b in mc.streaming_word_triples(trimmed, lang, tiny)]
        R['tsync_asr'] = arr(mc.kernel_frames(mc.timed_phonemes_from_triples(tr, lang), lang, n), n); meta['words_asr'] = tr
        t0 = time.perf_counter()
        ctc, lat, comp = run_ctc(al, audio, wj['words'], 0.16, 0.10, sil=0.20, avail=wj['avail'])
        meta['ctc_wall'] = time.perf_counter() - t0
        meta['words_ctc'] = [(norm_word(w), s, e) for w, s, e in ctc]; meta['ctc_lat_ms'] = [x * 1000 for x in lat]
        meta['ctc_comp_ms'] = [x * 1000 for x in comp]
        R['tsync_ctc'] = arr(mc.kernel_frames(mc.timed_phonemes_from_triples(meta['words_ctc'], lang), lang, n), n)
        R['mfa'] = arr(mc.kernel_frames(mfa_timed(tg, lang, mc), lang, n), n)
        if True:
            Wa = a2f.run(audio.astype(np.float32))
            t30 = np.arange(len(Wa)) / A2F_FPS; t60 = np.arange(n) / 60.0
            R['a2f'] = np.stack([np.interp(t60, t30, Wa[:, c]) for c in a2f_cols], 1).astype(np.float32)
        np.savez_compressed(out, **R)
        json.dump(meta, open(CD / f'{uid}.meta.json', 'w'), ensure_ascii=False)
        print(uid, 'ok', flush=True)

elif stage == 'eval':
    from scipy import stats
    import make_conditions as mc
    from compute_metrics import load_lip_deltas
    from grid_layer_validation import parse_textgrid_tier
    from faster_whisper.audio import decode_audio
    CH = list(mc.CHANNELS)
    lip, _ = load_lip_deltas(ROOT / 'atlas/public/human.glb')
    Dm = np.stack([lip[c] for c in CH])
    mesh = lambda B: np.einsum('tc,ckd->tkd', B, Dm)
    man = json.load(open(D / 'manifest.json'))
    uids = sorted(p.stem for p in CD.glob('*.npz'))
    conds = [c for c in ['amp', 'rate', 'tsync_small', 'tsync_asr', 'tsync_ctc', 'a2f'] if c in np.load(CD / f'{uids[0]}.npz')]
    rng = np.random.default_rng(0)
    boot = lambda x: np.percentile([rng.choice(x, len(x)).mean() for _ in range(5000)], [2.5, 97.5])
    res = {'lang': lang, 'n_utt': len(uids)}
    # utterance statistics (E1)
    dur = np.array([man[u]['duration_s'] for u in uids])
    nw_ = np.array([len(json.load(open(W16 / f'{u}.words.json'))['words']) for u in uids])
    res['E1'] = {'dur_mean': dur.mean(), 'dur_sd': dur.std(ddof=1), 'dur_min': dur.min(), 'dur_max': dur.max(),
                 'dur_total': dur.sum(), 'words_total': int(nw_.sum()), 'words_mean': nw_.mean()}
    # text lead: first text chunk vs first audio chunk (wall clock)
    lead = []
    for u in uids:
        ev = json.load(open(D / f'{u}.events.json'))
        ta = next(t for t, k, *_ in ev if k == 'audio'); tt = next((t for t, k, *_ in ev if k == 'text'), None)
        if tt is not None:
            lead.append((tt - ta) * 1000)
    res['text_minus_audio_ms'] = {'mean': float(np.mean(lead)), 'median': float(np.median(lead)), 'sd': float(np.std(lead))}
    # LVD to MFA mesh and envelope lag (paper's Table 4 metrics)
    L = {c: [] for c in conds}; LAG = {c: [] for c in conds}
    for u in uids:
        z = np.load(CD / f'{u}.npz'); ref = mesh(z['mfa'])
        audio = decode_audio(str(W16 / f'{u}.wav'), sampling_rate=16000)
        env = mc.rms_envelope(audio); env = env - env.mean()
        for c in conds:
            P = z[c][:len(ref)]
            L[c].append(float(np.linalg.norm(mesh(P) - ref[:len(P)], axis=2).mean()))
            jaw = P[:, 0] - P[:, 0].mean(); e = env[:len(jaw)]
            best, bl = -np.inf, 0
            for lag in range(-30, 31):
                a, b = (jaw[lag:], e[:len(e) - lag]) if lag >= 0 else (jaw[:lag], e[-lag:])
                if len(a) > 10 and a.std() > 0 and b.std() > 0:
                    r = np.corrcoef(a, b)[0, 1]
                    if r > best:
                        best, bl = r, lag
            LAG[c].append(bl / 60 * 1000)
    res['lvd'] = {c: {'mean': float(np.mean(v)), 'ci': boot(np.array(v)).tolist(), 'sd': float(np.std(v))} for c, v in L.items()}
    res['lag'] = {c: {'mean': float(np.mean(v)), 'sd': float(np.std(v))} for c, v in LAG.items()}
    ref_c = 'tsync_ctc'
    res['lvd_paired_vs_ctc'] = {c: {'delta': float(np.mean(np.array(L[ref_c]) - np.array(L[c]))),
                                     'p': float(stats.wilcoxon(np.array(L[ref_c]), np.array(L[c])).pvalue)}
                                for c in conds if c != ref_c}
    # content-matched word timing vs MFA (fixes the order-paired scoring of compute_metrics.py)
    tim = {}
    for key in ['words_small', 'words_asr', 'words_ctc']:
        st, en, per_off, resid, matched, total = [], [], [], [], 0, 0
        for u in uids:
            meta = json.load(open(CD / f'{u}.meta.json'))
            ref = [(norm_word(w), a, b) for w, a, b in parse_textgrid_tier(MO / f'{u}.TextGrid', 'words') if norm_word(w)]
            hyp = [(norm_word(w), a, b) for w, a, b in meta[key] if norm_word(w)]
            sm = difflib.SequenceMatcher(a=[x[0] for x in ref], b=[x[0] for x in hyp], autojunk=False)
            e_u = []
            for blk in sm.get_matching_blocks():
                for i in range(blk.size):
                    r, h = ref[blk.a + i], hyp[blk.b + i]
                    e_u.append((h[1] - r[1]) * 1000); en.append(abs(h[2] - r[2]) * 1000)
            matched += len(e_u); total += len(ref)
            if e_u:
                st += e_u; per_off.append(np.median(e_u)); resid += list(np.array(e_u) - np.median(e_u))
        s_ = np.array(st); off = np.median(s_)
        tim[key] = {'n': len(s_), 'match_rate': matched / max(1, total), 'start_mae': float(np.abs(s_).mean()),
                    'start_median_abs': float(np.median(np.abs(s_))), 'end_mae': float(np.mean(en)),
                    'signed_median': float(off), 'signed_iqr': np.percentile(s_, [25, 75]).tolist(),
                    'within_100ms': float(np.mean(np.abs(s_) < 100)),
                    'mae_after_global_comp': float(np.abs(s_ - off).mean()),
                    'per_utt_offset_median': float(np.median(per_off)), 'within_utt_resid_mae': float(np.abs(resid).mean())}
    res['timing_vs_mfa'] = tim
    # Layer-1 latency of the live streaming aligner
    lat = np.concatenate([json.load(open(CD / f'{u}.meta.json'))['ctc_lat_ms'] for u in uids])
    comp = np.concatenate([json.load(open(CD / f'{u}.meta.json'))['ctc_comp_ms'] for u in uids])
    res['ctc_latency_ms'] = {'median': float(np.median(lat)), 'p95': float(np.percentile(lat, 95)), 'max': float(lat.max()),
                             'comp_median': float(np.median(comp)), 'comp_p95': float(np.percentile(comp, 95)), 'n': int(len(lat))}
    json.dump(res, open(D / f'results_{lang}.json', 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
