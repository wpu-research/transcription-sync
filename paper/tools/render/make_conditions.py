"""
Section V-C condition generators: wav (+transcript) -> 60 fps 14-channel ARKit
blendshape schedule JSON for each evaluation condition (Section V-A):

  AMP         amplitude-only: smoothed RMS envelope -> JawOpen, rest at zero
  RATE        deployed rate-scheduled variant (Section IV-F): faithful offline
              port of main.py's _viseme_worker (13 chars/s, per-char viseme
              targets, RMS amplitude resynchronization, EMA alpha=0.40,
              inter-word closure poses)
  TSYNC       full TranscriptionSync: CAUSAL streaming Layer-1 simulation
              (sliding window W=3.2s, hop 320ms, guard 400ms — the protocol of
              Section IV-B with an idealized real-time hop) -> Layer 2 (Eq.4-5)
              -> Layer 3 (Eq.6-7)
  TSYNC-dur   TSYNC with uniform within-word durations (d_j = T_i/J)
  TSYNC-coart TSYNC with nearest-phoneme step targets instead of Eq.6 kernels
  MFA         non-causal oracle: phoneme timing from an MFA TextGrid drives
              the same viseme table and kernel synthesis

Usage:
  python make_conditions.py --wav utt.wav --lang en --outdir out/ \
      [--transcript "exact spoken text"] [--textgrid utt.TextGrid] \
      [--conditions amp,rate,tsync,tsync-dur,tsync-coart,mfa] [--asr-model small]

If --transcript is omitted, the ASR transcript is used for RATE (matching the
live system, which feeds Gemini's output transcription). MFA requires
--textgrid. Outputs <outdir>/<wavstem>_<condition>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # paper/tools
from layer23_bench import (ARPABET_TABLE, TURKISH_TABLE, BILABIAL_PHONES_EN,
                            BILABIAL_PHONES_TR, g2p_en, g2p_tr,
                            layer2_durations, layer3_render, CHANNELS,
                            SIGMA, SUPPORT, _GAP)

SR = 16000
FPS = 60
REST = {c: 0.0 for c in CHANNELS}
REST['MouthClose'] = 0.15

# Streaming Layer-1 protocol design values (Section IV-B)
WIN, HOP, GUARD = 3.2, 0.32, 0.40


def lang_tables(lang):
    if lang == 'en':
        return ARPABET_TABLE, BILABIAL_PHONES_EN, g2p_en
    return TURKISH_TABLE, BILABIAL_PHONES_TR, g2p_tr


def n_frames_for(audio) -> int:
    return round(len(audio) / SR * FPS)


def pad_to(frames, n_total):
    frames = list(frames)[:n_total]
    return frames + [dict(REST)] * max(0, n_total - len(frames))


def rms_envelope(audio, win_ms=50.0, attack=0.55, release=0.12):
    """Smoothed per-output-frame RMS envelope in [0,1] (95th-pct normalized)."""
    n_total = n_frames_for(audio)
    half = int(SR * win_ms / 1000.0 / 2)
    env = np.zeros(n_total)
    for f in range(n_total):
        c = int(f / FPS * SR)
        seg = audio[max(0, c - half):c + half]
        env[f] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
    norm = np.percentile(env[env > 0], 95) if (env > 0).any() else 1.0
    env = np.clip(env / max(norm, 1e-6), 0.0, 1.0)
    out = np.zeros_like(env)
    s = 0.0
    for i, v in enumerate(env):  # asymmetric attack/release smoothing
        s += (attack if v > s else release) * (v - s)
        out[i] = s
    return out


# ── AMP ────────────────────────────────────────────────────────────────────
def amp_schedule(audio):
    env = rms_envelope(audio)
    frames = []
    for v in env:
        fr = dict(REST)
        fr['MouthClose'] = 0.0
        fr['JawOpen'] = float(0.55 * v)
        frames.append(fr)
    return frames


# ── RATE (faithful port of main.py _viseme_worker) ─────────────────────────
# Viseme table copied verbatim from main.py (grapheme-class, 14 channels).
_VISEME = {
    'a': {'JawOpen': 0.60, 'MouthLowerDownLeft': 0.35, 'MouthLowerDownRight': 0.35, 'MouthUpperUpLeft': 0.18, 'MouthUpperUpRight': 0.18},
    'e': {'JawOpen': 0.32, 'MouthStretchLeft': 0.42, 'MouthStretchRight': 0.42, 'MouthLowerDownLeft': 0.18, 'MouthLowerDownRight': 0.18},
    'i': {'JawOpen': 0.18, 'MouthStretchLeft': 0.52, 'MouthStretchRight': 0.52},
    'o': {'JawOpen': 0.42, 'MouthFunnel': 0.38, 'MouthPucker': 0.18, 'MouthLowerDownLeft': 0.22, 'MouthLowerDownRight': 0.22},
    'u': {'JawOpen': 0.18, 'MouthPucker': 0.58, 'MouthFunnel': 0.42},
    'm': {'JawOpen': 0.00, 'MouthClose': 0.92, 'MouthShrugUpper': 0.15},
    'b': {'JawOpen': 0.04, 'MouthClose': 0.65},
    'p': {'JawOpen': 0.02, 'MouthClose': 0.80},
    'f': {'JawOpen': 0.06, 'MouthUpperUpLeft': 0.58, 'MouthUpperUpRight': 0.58, 'MouthLowerDownLeft': 0.12, 'MouthLowerDownRight': 0.12},
    'v': {'JawOpen': 0.08, 'MouthUpperUpLeft': 0.48, 'MouthUpperUpRight': 0.48},
    's': {'JawOpen': 0.07, 'MouthStretchLeft': 0.28, 'MouthStretchRight': 0.28},
    'z': {'JawOpen': 0.10, 'MouthStretchLeft': 0.22, 'MouthStretchRight': 0.22},
    'w': {'JawOpen': 0.14, 'MouthPucker': 0.50, 'MouthFunnel': 0.38},
    'r': {'JawOpen': 0.20, 'MouthFunnel': 0.18, 'MouthLowerDownLeft': 0.14, 'MouthLowerDownRight': 0.14},
    'l': {'JawOpen': 0.16, 'MouthLowerDownLeft': 0.10, 'MouthLowerDownRight': 0.10},
    't': {'JawOpen': 0.10, 'MouthStretchLeft': 0.12, 'MouthStretchRight': 0.12},
    'd': {'JawOpen': 0.13, 'MouthStretchLeft': 0.10, 'MouthStretchRight': 0.10},
    'n': {'JawOpen': 0.10},
    'h': {'JawOpen': 0.28},
    'j': {'JawOpen': 0.14, 'MouthStretchLeft': 0.18, 'MouthStretchRight': 0.18},
    'k': {'JawOpen': 0.20}, 'g': {'JawOpen': 0.22}, 'x': {'JawOpen': 0.15},
    'c': {'JawOpen': 0.10, 'MouthStretchLeft': 0.15, 'MouthStretchRight': 0.15},
    'q': {'JawOpen': 0.18}, 'y': {'JawOpen': 0.14, 'MouthStretchLeft': 0.20, 'MouthStretchRight': 0.20},
    '_': {'JawOpen': 0.12, 'MouthShrugUpper': 0.08},
}
_WORD_PAUSE = {'JawOpen': 0.04, 'MouthClose': 0.28}
_CHARS_PER_SEC = 13.0
_RATE_ALPHA = 0.40


def rate_schedule(audio, transcript: str):
    """Offline port of the deployed rate-scheduled variant. The live worker
    advances one character per 1/13 s tick and applies EMA smoothing; the
    character clock starts when the first transcription chunk arrives, which
    coincides with audio onset — emulated here as the first frame where the
    RMS envelope exceeds 10% (same onset logic class as Section V-D)."""
    env = rms_envelope(audio)
    n_total = n_frames_for(audio)

    tokens = []
    for c in transcript.lower():
        if c.isalpha():
            tokens.append(('char', c))
        elif c in (' ', ',', '.', '!', '?', ';', ':'):
            tokens.append(('pause', c))
    if not tokens:
        return [dict(REST)] * n_total

    onset_f = int(np.argmax(env > 0.10)) if (env > 0.10).any() else 0
    char_dur_f = FPS / _CHARS_PER_SEC  # frames per character tick

    frames = []
    current = {k: 0.0 for k in CHANNELS}
    for f in range(n_total):
        idx = int((f - onset_f) / char_dur_f)
        if f < onset_f or idx >= len(tokens):
            target = dict(_WORD_PAUSE)
        else:
            kind, ch = tokens[idx]
            target = dict(_WORD_PAUSE if kind == 'pause' else _VISEME.get(ch, _VISEME['_']))
        # amplitude resynchronization (live: amp = min(1.4, jaw_smooth*2.2))
        amp = min(1.4, float(env[f]) * 2.2)
        scaled = {k: min(1.0, v * amp) for k, v in target.items()}
        if 'MouthClose' not in scaled:
            scaled['MouthClose'] = max(0.0, 0.08 - scaled.get('JawOpen', 0) * 0.2)
        # live EMA runs once per char tick; per frame the equivalent factor is
        # alpha_f = 1-(1-alpha)^(1/char_dur_f)
        alpha_f = 1.0 - (1.0 - _RATE_ALPHA) ** (1.0 / char_dur_f)
        for k in CHANNELS:
            current[k] += alpha_f * (scaled.get(k, 0.0) - current[k])
        frames.append(dict(current))
    return frames


# ── Layer 1: causal streaming word-triple extraction ───────────────────────
def streaming_word_triples(audio, lang, model):
    """Sliding-window finalization protocol of Section IV-B (idealized
    real-time: window advances by exactly HOP). Returns finalized
    (text, t_start, t_end) triples on the stream clock — deterministic and
    machine-independent (wall-clock cost is Section V-E's concern, not here)."""
    duration = len(audio) / SR
    window_end = min(WIN, duration)
    finalized = []
    while True:
        window_start = max(0.0, window_end - WIN)
        seg = audio[int(window_start * SR):int(window_end * SR)]
        segments, _ = model.transcribe(seg, word_timestamps=True, language=lang,
                                         vad_filter=False, condition_on_previous_text=False)
        for s in segments:
            for w in (s.words or []):
                text = w.word.strip().lower().strip('.,!?')
                ts, te = window_start + w.start, window_start + w.end
                if (window_end - te) < GUARD and window_end < duration:
                    continue  # may still be revised by a later window
                if any(text == ft and abs(te - fte) < 0.12 for ft, _, fte in finalized):
                    continue
                finalized.append((text, ts, te))
        if window_end >= duration:
            break
        window_end = min(duration, window_end + HOP)
    finalized.sort(key=lambda x: x[1])
    return finalized


# ── TSYNC and ablations ─────────────────────────────────────────────────────
def timed_phonemes_from_triples(triples, lang, uniform=False):
    table, bilset, g2p = lang_tables(lang)
    timed = []
    for word, t_s, t_e in triples:
        phonemes = [p for p in g2p(word) if p in table]
        if not phonemes:
            continue
        if uniform:
            J = len(phonemes)
            durs = [(t_e - t_s) / J] * J
            mids, acc = [], 0.0
            for d in durs:
                mids.append(t_s + acc + d / 2.0)
                acc += d
        else:
            _, mids = layer2_durations(phonemes, t_s, t_e, table)
        for p, tau in zip(phonemes, mids):
            cls, V = table[p]
            timed.append((p, tau, V, p in bilset))
    timed.sort(key=lambda x: x[1])
    return timed


def kernel_frames(timed, lang, n_total):
    table, bilset, _ = lang_tables(lang)
    frames, _t = layer3_render(timed, table, bilset, fps=FPS)
    if not timed:
        return [dict(REST)] * n_total
    t0 = timed[0][1] - SUPPORT
    lead = max(0, round(t0 * FPS))
    return pad_to([dict(REST)] * lead + frames, n_total)


def step_frames(timed, n_total):
    """TSYNC-coart ablation: nearest-phoneme step targets (no Eq.6 kernel)."""
    frames = []
    for f in range(n_total):
        t = f / FPS
        local = [(abs(t - tau), V) for (_, tau, V, _) in timed if abs(t - tau) <= SUPPORT]
        if not local:
            frames.append(dict(_GAP))
        else:
            frames.append(dict(min(local, key=lambda x: x[0])[1]))
    return frames


# ── MFA oracle ──────────────────────────────────────────────────────────────
def mfa_timed_phonemes(textgrid: Path, lang):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from grid_layer_validation import parse_textgrid_phones, clean_phone
    table, bilset, _ = lang_tables(lang)
    timed = []
    for p, a, b in parse_textgrid_phones(textgrid):
        ph = clean_phone(p)
        if not ph or ph not in table:
            continue
        tau = (a + b) / 2.0
        cls, V = table[ph]
        timed.append((ph, tau, V, ph in bilset))
    timed.sort(key=lambda x: x[1])
    return timed


# ── main ────────────────────────────────────────────────────────────────────
ALL_CONDITIONS = ['amp', 'rate', 'tsync', 'tsync-dur', 'tsync-coart', 'mfa']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--lang', required=True, choices=['en', 'tr'])
    ap.add_argument('--transcript', default=None)
    ap.add_argument('--textgrid', default=None)
    ap.add_argument('--asr-model', default='small')
    ap.add_argument('--conditions', default=','.join(ALL_CONDITIONS))
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    wav = Path(args.wav)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audio = decode_audio(str(wav), sampling_rate=SR)
    n_total = n_frames_for(audio)
    conditions = [c.strip() for c in args.conditions.split(',')]

    triples = None
    model = None
    need_asr = any(c.startswith('tsync') for c in conditions) or \
               ('rate' in conditions and not args.transcript)
    if need_asr:
        model = WhisperModel(args.asr_model, device='cpu', compute_type='int8')
        triples = streaming_word_triples(audio, args.lang, model)
        print(f'  Layer-1 ({args.asr_model}, causal): '
              f'{" ".join(t for t, _, _ in triples)}')

    meta = {'fps': FPS, 'wav': wav.name, 'lang': args.lang}
    results = {}

    for cond in conditions:
        if cond == 'amp':
            frames = amp_schedule(audio)
        elif cond == 'rate':
            transcript = args.transcript or ' '.join(t for t, _, _ in (triples or []))
            frames = rate_schedule(audio, transcript)
        elif cond == 'tsync':
            frames = kernel_frames(timed_phonemes_from_triples(triples, args.lang), args.lang, n_total)
        elif cond == 'tsync-dur':
            frames = kernel_frames(timed_phonemes_from_triples(triples, args.lang, uniform=True), args.lang, n_total)
        elif cond == 'tsync-coart':
            timed = timed_phonemes_from_triples(triples, args.lang)
            frames = pad_to(step_frames(timed, n_total), n_total)
        elif cond == 'mfa':
            if not args.textgrid:
                print('  [skip] mfa: --textgrid not given')
                continue
            timed = mfa_timed_phonemes(Path(args.textgrid), args.lang)
            frames = kernel_frames(timed, args.lang, n_total)
        else:
            print(f'  [skip] unknown condition {cond}')
            continue
        out = outdir / f'{wav.stem}_{cond}.json'
        sched = dict(meta)
        if triples is not None:
            sched['words'] = [(w, round(s, 3), round(e, 3)) for w, s, e in triples]
        sched['condition'] = cond
        sched['frames'] = pad_to(frames, n_total)
        out.write_text(json.dumps(sched))
        results[cond] = out.name
    print(f'{wav.stem}: {len(results)} schedules -> {outdir} ({", ".join(results)})')


if __name__ == '__main__':
    main()
