"""
Layer-wise validation of TranscriptionSync on the GRID corpus (Section V-D,
Table 5 of the ATLAS IEEE Access draft).

This produces REAL measurements for the three `[MOCK]` rows of Table 5:
  (i)   Layer-1 word boundary error: faster-whisper "tiny" vs GRID .align word
        boundaries, + word error rate.
  (ii)  Layer-2 phoneme timing error in isolation: Eq. (4) class-prior duration
        allocation (DUR_PRIOR table, imported from layer23_bench.py — these are
        the FIXED design values, not re-fit per corpus) applied to GRID GT word
        boundaries, vs MFA phoneme midpoints. Uniform-split ablation included.
  (iii) Composed Layer-1+2 error: Whisper word boundaries + Eq.(4) priors, vs
        MFA phoneme midpoints.

Methodology note on (ii)/(iii): for a given word, MFA gives the REFERENCE
phoneme sequence with timing. We do NOT re-derive a phoneme sequence via G2P
for the timing comparison (G2P/MFA phone-count mismatches would confound the
"how to split a known sequence" question that Eq. 4 actually answers). Instead
we take MFA's phone sequence for the word as fixed, and compare two ways of
splitting the word's time span across exactly those phones: Eq.(4) class
priors vs uniform. This isolates Layer 2's actual job.

Requires (run on a machine with the GRID corpus already downloaded):
  - faster-whisper, cmudict, numpy  (pip install faster-whisper cmudict numpy)
  - Montreal Forced Aligner (conda install -c conda-forge montreal-forced-aligner)
    with: mfa model download acoustic english_us_arpa
          mfa model download dictionary english_us_arpa

Pipeline:
  1) python grid_layer_validation.py prepare-mfa  <grid_corpus> <mfa_corpus> --speakers s29-s34
  2) mfa align <mfa_corpus> english_us_arpa english_us_arpa <mfa_out> --clean
  3) python grid_layer_validation.py layer1       <grid_corpus> --speakers s29-s34 --out layer1_results.json
  4) python grid_layer_validation.py layer2       <grid_corpus> <mfa_out> --speakers s29-s34 --out layer2_results.json
  5) python grid_layer_validation.py report layer1_results.json layer2_results.json --out table5.md

Speakers are given as e.g. "s29-s34" (range) or a comma-separated list
"s29,s30,s31". n is reported honestly from however many utterances were
actually processed (some GRID speakers/utterances are missing/corrupt and are
skipped with a warning, not silently padded).
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer23_bench import ARPABET_TABLE, DUR_PRIOR, g2p_en  # noqa: E402

SR = 16000


# ── speaker list parsing ───────────────────────────────────────────────────
def parse_speakers(spec: str) -> list[str]:
    out = []
    for part in spec.split(','):
        part = part.strip()
        m = re.match(r's(\d+)-s(\d+)$', part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            out.extend(f's{i}' for i in range(lo, hi + 1))
        else:
            out.append(part)
    return out


# ── GRID .align parsing ────────────────────────────────────────────────────
def read_align(path: Path):
    """Returns list of (word, start_s, end_s), excluding 'sil'/'sp'."""
    words = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        start, end, w = parts
        if w in ('sil', 'sp'):
            continue
        words.append((w, int(start) / 25000.0, int(end) / 25000.0))
    return words


def find_utterances(grid_corpus: Path, speakers: list[str]):
    """Yield (speaker, utt_id, wav_path, align_path) for utterances that have
    both a .wav and a .align file."""
    for spk in speakers:
        spk_dir = grid_corpus / spk
        align_dir = spk_dir / 'align'
        # video/audio dir is sometimes nested as <spk>/<spk>/*.wav
        wav_candidates = list(spk_dir.glob(f'{spk}/*.wav')) or list(spk_dir.glob('*.wav')) \
            or list(spk_dir.glob('audio/*.wav'))
        wav_by_id = {p.stem: p for p in wav_candidates}
        if not align_dir.is_dir():
            print(f"  [skip] {spk}: no align/ directory", file=sys.stderr)
            continue
        for align_path in sorted(align_dir.glob('*.align')):
            utt_id = align_path.stem
            wav_path = wav_by_id.get(utt_id)
            if wav_path is None or not wav_path.exists():
                print(f"  [skip] {spk}/{utt_id}: no matching .wav", file=sys.stderr)
                continue
            yield spk, utt_id, wav_path, align_path


# ── Step 1: prepare MFA corpus ─────────────────────────────────────────────
def cmd_prepare_mfa(args):
    grid_corpus = Path(args.grid_corpus)
    mfa_corpus = Path(args.mfa_corpus)
    speakers = parse_speakers(args.speakers)
    n = 0
    for spk, utt_id, wav_path, align_path in find_utterances(grid_corpus, speakers):
        words = read_align(align_path)
        if not words:
            continue
        out_dir = mfa_corpus / spk
        out_dir.mkdir(parents=True, exist_ok=True)
        lab_path = out_dir / f'{utt_id}.lab'
        lab_path.write_text(' '.join(w for w, _, _ in words))
        wav_out = out_dir / f'{utt_id}.wav'
        if not wav_out.exists():
            try:
                os.symlink(wav_path.resolve(), wav_out)
            except OSError:
                import shutil
                shutil.copy(wav_path, wav_out)
        n += 1
    print(f"Prepared {n} utterances for MFA in {mfa_corpus}")
    print(f"Next: mfa align {mfa_corpus} english_us_arpa english_us_arpa <mfa_out> --clean")


# ── minimal TextGrid (Praat long-text format) parser ──────────────────────
def parse_textgrid_phones(path: Path):
    """Returns list of (phone, start_s, end_s) for the 'phones' tier."""
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    intervals = []
    in_phones = False
    cur = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('name = "phones"'):
            in_phones = True
            continue
        if s.startswith('name = "') and in_phones:
            in_phones = False
        if not in_phones:
            continue
        if s.startswith('xmin ='):
            cur['xmin'] = float(s.split('=')[1])
        elif s.startswith('xmax ='):
            cur['xmax'] = float(s.split('=')[1])
        elif s.startswith('text ='):
            txt = s.split('=', 1)[1].strip().strip('"')
            if 'xmin' in cur and 'xmax' in cur:
                intervals.append((txt, cur['xmin'], cur['xmax']))
            cur = {}
    return intervals


_MFA_PHONE_RE = re.compile(r'^([A-Za-z]+)\d*$')


def clean_phone(p: str) -> str:
    """Strip MFA stress digits; map silence/spoken-noise markers to None."""
    p = p.strip()
    if p in ('', 'sil', 'sp', 'spn', '<eps>'):
        return ''
    m = _MFA_PHONE_RE.match(p)
    return m.group(1).upper() if m else ''


def trim_leading_silence(audio: np.ndarray, sr: int = SR,
                          frame_ms: float = 10.0, rel_thresh: float = 0.10,
                          sustain_frames: int = 5,
                          margin_ms: float = 80.0) -> tuple[np.ndarray, float]:
    """Energy-based leading-silence trim. Returns (trimmed_audio, offset_s).

    Whisper's DTW anchors its clock at speech onset: on a clip with leading
    silence, ALL word timestamps come out shifted early by roughly the
    leading-silence duration (measured on GRID: mean -326 ms, 94% early).
    Silero VAD (vad_filter=True) does NOT fix this (default speech_pad_ms=400
    leaves most of GRID's 0.2-0.95 s leading silences in place). So we trim
    deterministically with a simple RMS threshold and add the offset back to
    the returned timestamps — transparent and reproducible.

    Onset requires SUSTAINED energy (sustain_frames consecutive frames above
    threshold, default 50 ms): GRID's lead-in contains single-frame clicks and
    breath spikes that exceed any per-frame threshold and would otherwise
    anchor the onset at ~0 (observed: trim_offset=0.00 on utterances whose
    speech starts at 0.4-0.7 s). Speech is sustained; clicks are not.

    A margin (default 80 ms) is left before the detected onset so that weak
    speech onsets (fricatives, breathy voicing) are not clipped.
    """
    frame = int(sr * frame_ms / 1000.0)
    if len(audio) < 2 * frame:
        return audio, 0.0
    n_frames = len(audio) // frame
    rms = np.sqrt((audio[:n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1))
    thresh = float(rms.max()) * rel_thresh
    if thresh <= 0:
        return audio, 0.0
    above = rms > thresh
    if len(above) < sustain_frames:
        return audio, 0.0
    # first index where `sustain_frames` consecutive frames are all above
    runs = np.convolve(above.astype(int), np.ones(sustain_frames, dtype=int), mode='valid')
    sustained = runs >= sustain_frames
    if not sustained.any():
        return audio, 0.0
    onset_frame = int(np.argmax(sustained))
    onset_s = max(0.0, onset_frame * frame_ms / 1000.0 - margin_ms / 1000.0)
    start = int(onset_s * sr)
    return audio[start:], onset_s


_NUM_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
}
_LETTER_DIGIT_RE = re.compile(r'^([a-z])(\d)$')


def normalize_wer_tokens(tokens: list[str]) -> list[str]:
    """Map spoken digit words to digit characters, and split merged
    'letter+digit' tokens (Whisper's "c4" for GRID's "c four") back into two
    tokens, so WER reflects recognition errors rather than tokenization
    differences for GRID's <letter><digit> callsign vocabulary."""
    out = []
    for t in tokens:
        if t in _NUM_WORDS:
            out.append(_NUM_WORDS[t])
            continue
        m = _LETTER_DIGIT_RE.match(t)
        if m:
            out.append(m.group(1))
            out.append(m.group(2))
            continue
        out.append(t)
    return out


# ── Step 3: Layer-1 (Whisper word boundaries vs GRID GT) ───────────────────
def align_words_lev(ref: list[str], hyp: list[str]):
    """Levenshtein alignment; returns list of (ref_idx_or_None, hyp_idx_or_None)
    operation pairs, plus the edit distance (substitutions+ins+del)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    # backtrack
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1  # substitution, not used for MAE
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    pairs.reverse()
    return pairs, dp[n][m]


def cmd_layer1(args):
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    grid_corpus = Path(args.grid_corpus)
    speakers = parse_speakers(args.speakers)
    print(f"Loading faster-whisper '{args.model}' model (may download on first run)...", file=sys.stderr)
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print("Model loaded. Starting transcription loop...", file=sys.stderr)

    # Separate error series for start / midpoint / end of each matched word:
    # Whisper end timestamps are known to be biased late (words extended into
    # silence), so the three series diagnose where the boundary error lives.
    errs_start_ms, errs_mid_ms, errs_end_ms = [], [], []
    total_ref_words = 0
    total_edits = 0
    n_utt = 0
    n_correct_match = 0

    import time
    t_start = time.time()
    n_skipped_tg = 0
    for spk, utt_id, wav_path, align_path in find_utterances(grid_corpus, speakers):
        # Word-boundary reference: MFA TextGrid 'words' tier, NOT the
        # corpus-supplied .align. The distributed audio edition we use was
        # found temporally inconsistent with the .align files (speech starts
        # ~0.04 s into the wav, while .align assumes 0.2-0.95 s of leading
        # silence; per-utterance shift mean -362 ms, sd 165 ms over a
        # 50-utterance check — MFA and Whisper independently agree on the
        # audio clock). MFA operates near ceiling on studio-quality read
        # speech and is already the phoneme-level reference (Section V-D).
        if args.mfa_out:
            tg = Path(args.mfa_out) / spk / f'{utt_id}.TextGrid'
            if not tg.exists():
                n_skipped_tg += 1
                continue
            ref_words = parse_textgrid_tier(tg, 'words')
        else:
            ref_words = read_align(align_path)
        if not ref_words:
            continue
        audio = decode_audio(str(wav_path), sampling_rate=SR)
        # Trim is a no-op on this audio edition (speech starts ~0.04 s) but
        # kept for generality; offset is restored to the timestamps.
        audio_trimmed, offset_s = trim_leading_silence(audio)
        segments, _ = model.transcribe(audio_trimmed, word_timestamps=True, language="en",
                                         vad_filter=False, condition_on_previous_text=False)
        hyp = [(w.word.strip().lower().strip('.,!?'), w.start + offset_s, w.end + offset_s)
               for s in segments for w in (s.words or [])]
        ref_tokens = [w for w, _, _ in ref_words]
        hyp_tokens = [w for w, _, _ in hyp]

        # WER is computed on normalized token streams: GRID's spoken digit
        # words ("four") are mapped to digit characters ("4") in the
        # reference, and Whisper's merged "letter+digit" callsign tokens
        # ("c4") are split back into two tokens ("c","4") in the hypothesis.
        # Without this, every "<letter> <digit>" pair (2/6 words of every
        # GRID sentence) registers as 2 spurious errors purely from
        # tokenization, not recognition.
        ref_norm = normalize_wer_tokens(ref_tokens)
        hyp_norm = normalize_wer_tokens(hyp_tokens)
        _, edits = align_words_lev(ref_norm, hyp_norm)
        total_ref_words += len(ref_norm)
        total_edits += edits

        # Boundary MAE uses the ORIGINAL (unsplit) token alignment: a merged
        # "c4" hypothesis token cannot be matched to either "c" or "four", so
        # those words simply contribute no boundary sample (lower coverage,
        # but no fabricated timestamps).
        pairs, _ = align_words_lev(ref_tokens, hyp_tokens)

        for ri, hi in pairs:
            if ri is None or hi is None:
                continue
            if ref_tokens[ri] != hyp_tokens[hi]:
                continue
            _, ref_s, ref_e = ref_words[ri]
            _, hyp_s, hyp_e = hyp[hi]
            errs_start_ms.append(abs(hyp_s - ref_s) * 1000.0)
            errs_mid_ms.append(abs((hyp_s + hyp_e) / 2 - (ref_s + ref_e) / 2) * 1000.0)
            errs_end_ms.append(abs(hyp_e - ref_e) * 1000.0)
            n_correct_match += 1
        n_utt += 1
        if n_utt % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  [{n_utt} utterances, {elapsed:.0f}s elapsed, "
                  f"{elapsed / n_utt:.2f}s/utt]", file=sys.stderr)

    def mae_sd(arr):
        a = np.array(arr)
        if len(a) == 0:
            return None, None
        return float(np.mean(a)), float(np.std(a))

    mae_s, sd_s = mae_sd(errs_start_ms)
    mae_m, sd_m = mae_sd(errs_mid_ms)
    mae_e, sd_e = mae_sd(errs_end_ms)
    result = {
        'model': args.model,
        'reference': 'mfa_words_tier' if args.mfa_out else 'grid_align',
        'n_utterances': n_utt,
        'n_textgrids_missing': n_skipped_tg,
        'n_ref_words': total_ref_words,
        'n_matched_words': len(errs_end_ms),
        'wer_pct': 100.0 * total_edits / max(1, total_ref_words),
        'word_start_mae_ms': mae_s, 'word_start_sd_ms': sd_s,
        'word_mid_mae_ms': mae_m, 'word_mid_sd_ms': sd_m,
        'word_end_mae_ms': mae_e, 'word_end_sd_ms': sd_e,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


# ── Step 4: Layer-2 (Eq.4 priors vs MFA, on GRID GT word boundaries) ───────
def lambda_for_phone(phone: str) -> float:
    entry = ARPABET_TABLE.get(phone)
    if entry is None:
        return 1.0  # neutral fallback for phones outside our 14-channel table (e.g. unexpected)
    cls = entry[0]
    return DUR_PRIOR[cls]


def split_with_lambdas(t_start, t_end, lambdas):
    total = sum(lambdas)
    T = t_end - t_start
    durations = [T * lam / total for lam in lambdas]
    midpoints, acc = [], 0.0
    for d in durations:
        midpoints.append(t_start + acc + d / 2.0)
        acc += d
    return midpoints


def cmd_layer2(args):
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    grid_corpus = Path(args.grid_corpus)
    mfa_out = Path(args.mfa_out)
    speakers = parse_speakers(args.speakers)

    use_asr = args.with_asr
    model = WhisperModel(args.model, device="cpu", compute_type="int8") if use_asr else None

    err_prior, err_uniform, err_composed = [], [], []
    n_words, n_utt, n_skipped_tg = 0, 0, 0

    for spk, utt_id, wav_path, align_path in find_utterances(grid_corpus, speakers):
        tg_path = mfa_out / spk / f'{utt_id}.TextGrid'
        if not tg_path.exists():
            n_skipped_tg += 1
            continue
        # GT word boundaries from the MFA 'words' tier, NOT the corpus .align:
        # the audio edition is temporally inconsistent with .align (see
        # cmd_layer1 comment); the MFA words tier is on the same clock as the
        # phones tier, so phone-into-word grouping below is exact.
        gt_words = parse_textgrid_tier(tg_path, 'words')
        if not gt_words:
            continue
        phones = parse_textgrid_phones(tg_path)
        # group phone intervals into words by GT word time spans
        hyp_words = None
        if use_asr:
            audio = decode_audio(str(wav_path), sampling_rate=SR)
            # See trim_leading_silence: deterministic onset trim + clock restore.
            audio_trimmed, offset_s = trim_leading_silence(audio)
            segments, _ = model.transcribe(audio_trimmed, word_timestamps=True, language="en",
                                            vad_filter=False, condition_on_previous_text=False)
            hyp_words = [(w.word.strip().lower().strip('.,!?'), w.start + offset_s, w.end + offset_s)
                          for s in segments for w in (s.words or [])]

        for wi, (word, ws, we) in enumerate(gt_words):
            # MFA phones whose midpoint falls within [ws, we)
            word_phones = [(clean_phone(p), a, b) for (p, a, b) in phones
                            if clean_phone(p) and ws <= (a + b) / 2.0 < we]
            if len(word_phones) < 1:
                continue
            mfa_midpoints = [(a + b) / 2.0 for (_, a, b) in word_phones]
            lambdas = [lambda_for_phone(p) for (p, _, _) in word_phones]

            pred_prior = split_with_lambdas(ws, we, lambdas)
            pred_uniform = split_with_lambdas(ws, we, [1.0] * len(word_phones))
            err_prior.extend(abs(p - m) * 1000.0 for p, m in zip(pred_prior, mfa_midpoints))
            err_uniform.extend(abs(p - m) * 1000.0 for p, m in zip(pred_uniform, mfa_midpoints))
            n_words += 1

            if use_asr:
                # find a matching ASR word by token + nearest overlap
                cands = [h for h in hyp_words if h[0] == word]
                if not cands:
                    continue
                hw = min(cands, key=lambda h: abs(h[1] - ws) + abs(h[2] - we))
                pred_composed = split_with_lambdas(hw[1], hw[2], lambdas)
                err_composed.extend(abs(p - m) * 1000.0 for p, m in zip(pred_composed, mfa_midpoints))
        n_utt += 1

    def stats(arr):
        a = np.array(arr)
        if len(a) == 0:
            return None, None, 0
        return float(np.mean(a)), float(np.std(a)), len(a)

    mae_p, sd_p, n_p = stats(err_prior)
    mae_u, sd_u, n_u = stats(err_uniform)
    mae_c, sd_c, n_c = stats(err_composed)

    result = {
        'n_utterances': n_utt,
        'n_textgrids_missing': n_skipped_tg,
        'n_words': n_words,
        'phoneme_midpoint_mae_ms_gt_words_lambda_priors': mae_p,
        'phoneme_midpoint_sd_ms_gt_words_lambda_priors': sd_p,
        'phoneme_midpoint_mae_ms_gt_words_uniform': mae_u,
        'phoneme_midpoint_sd_ms_gt_words_uniform': sd_u,
        'n_phones_compared': n_p,
        'phoneme_midpoint_mae_ms_asr_words_lambda_priors': mae_c,
        'phoneme_midpoint_sd_ms_asr_words_lambda_priors': sd_c,
        'n_phones_compared_composed': n_c,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


# ── Diagnostic: MFA word boundaries vs GRID .align (corpus consistency) ────
def parse_textgrid_tier(path: Path, tier: str):
    """Returns list of (text, start_s, end_s) for the named TextGrid tier."""
    lines = path.read_text(encoding='utf-8').splitlines()
    out, inside, cur = [], False, {}
    for line in lines:
        s = line.strip()
        if s.startswith('name = '):
            inside = (s == f'name = "{tier}"')
            continue
        if not inside:
            continue
        if s.startswith('xmin ='):
            cur['a'] = float(s.split('=')[1])
        elif s.startswith('xmax ='):
            cur['b'] = float(s.split('=')[1])
        elif s.startswith('text ='):
            t = s.split('=', 1)[1].strip().strip('"')
            if t and 'a' in cur:
                out.append((t.lower(), cur['a'], cur['b']))
            cur = {}
    return out


def cmd_check_align(args):
    """Referee test: if MFA word starts also show a systematic shift vs the
    GRID .align word starts, the corpus audio and .align files are mutually
    shifted (and Whisper's measured -325 ms shift is a corpus artifact, not
    an ASR error). If MFA agrees with .align (within ~±30 ms), the shift is
    genuinely Whisper's."""
    grid_corpus = Path(args.grid_corpus)
    mfa_out = Path(args.mfa_out)
    speakers = parse_speakers(args.speakers)
    diffs = []
    shown = 0
    n_utt = 0
    for spk, utt_id, wav_path, align_path in find_utterances(grid_corpus, speakers):
        tg = mfa_out / spk / f'{utt_id}.TextGrid'
        if not tg.exists():
            continue
        ref = read_align(align_path)
        mfa = parse_textgrid_tier(tg, 'words')
        if shown < 3:
            print(f'{utt_id}:')
            print('  ALIGN:', [(w, round(s, 2)) for w, s, e in ref])
            print('  MFA  :', [(w, round(s, 2)) for w, s, e in mfa])
            shown += 1
        rd = {w: s for w, s, e in ref}
        for w, s, e in mfa:
            if w in rd:
                diffs.append((s - rd[w]) * 1000.0)
        n_utt += 1
        if args.limit and n_utt >= args.limit:
            break
    d = np.array(diffs)
    print(f'n_utt={n_utt}  n_words={len(d)}')
    print(f'MFA-vs-ALIGN word start: mean={d.mean():.0f}ms  median={np.median(d):.0f}ms  '
          f'sd={d.std():.0f}ms  MAE={np.abs(d).mean():.0f}ms')


# ── Step 5: assemble Table 5 ────────────────────────────────────────────────
def cmd_report(args):
    l1 = json.loads(Path(args.layer1).read_text())
    l2 = json.loads(Path(args.layer2).read_text())

    def fmt(mae, sd):
        if mae is None:
            return "—"
        return f"{mae:.0f} ({sd:.0f})" if sd is not None else f"{mae:.0f}"

    lines = []
    lines.append(f"n utterances (Layer 1): {l1['n_utterances']}, matched words: {l1['n_matched_words']}/{l1['n_ref_words']}")
    lines.append(f"n utterances (Layer 2): {l2['n_utterances']}, words compared: {l2['n_words']} ({l2['n_phones_compared']} phones)")
    lines.append("")
    lines.append("| Quantity | Layer | Value |")
    lines.append("|---|---|---|")
    lines.append(f"| Word start MAE, Whisper vs GRID GT (ms) | 1 | {fmt(l1['word_start_mae_ms'], l1['word_start_sd_ms'])} |")
    lines.append(f"| Word midpoint MAE, Whisper vs GRID GT (ms) | 1 | {fmt(l1['word_mid_mae_ms'], l1['word_mid_sd_ms'])} |")
    lines.append(f"| Word end MAE, Whisper vs GRID GT (ms) | 1 | {fmt(l1['word_end_mae_ms'], l1['word_end_sd_ms'])} |")
    lines.append(f"| Word error rate (%) | 1 | {l1['wer_pct']:.1f} |")
    lines.append(f"| Phoneme midpoint MAE, GT words + λ priors (ms) | 2 | "
                  f"{fmt(l2['phoneme_midpoint_mae_ms_gt_words_lambda_priors'], l2['phoneme_midpoint_sd_ms_gt_words_lambda_priors'])} |")
    lines.append(f"| Phoneme midpoint MAE, GT words + uniform split (ms) | 2 | "
                  f"{fmt(l2['phoneme_midpoint_mae_ms_gt_words_uniform'], l2['phoneme_midpoint_sd_ms_gt_words_uniform'])} |")
    if l2.get('phoneme_midpoint_mae_ms_asr_words_lambda_priors') is not None:
        lines.append(f"| Phoneme midpoint MAE, ASR words + λ priors (ms) | 1+2 | "
                      f"{fmt(l2['phoneme_midpoint_mae_ms_asr_words_lambda_priors'], l2['phoneme_midpoint_sd_ms_asr_words_lambda_priors'])} |")
    else:
        lines.append("| Phoneme midpoint MAE, ASR words + λ priors (ms) | 1+2 | — (run layer2 with --with-asr) |")

    out = '\n'.join(lines)
    print(out)
    if args.out:
        Path(args.out).write_text(out, encoding='utf-8')


def main():
    # Windows consoles default to cp1252, which cannot encode λ etc.
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    pm = sub.add_parser('prepare-mfa')
    pm.add_argument('grid_corpus')
    pm.add_argument('mfa_corpus')
    pm.add_argument('--speakers', required=True)
    pm.set_defaults(func=cmd_prepare_mfa)

    p1 = sub.add_parser('layer1')
    p1.add_argument('grid_corpus')
    p1.add_argument('--speakers', required=True)
    p1.add_argument('--model', default='tiny', help="faster-whisper model size: tiny/base/small")
    p1.add_argument('--mfa-out', default=None,
                    help="use MFA TextGrid 'words' tier as the boundary reference "
                         "(required when the corpus audio edition is inconsistent with .align)")
    p1.add_argument('--out', default=None)
    p1.set_defaults(func=cmd_layer1)

    p2 = sub.add_parser('layer2')
    p2.add_argument('grid_corpus')
    p2.add_argument('mfa_out')
    p2.add_argument('--speakers', required=True)
    p2.add_argument('--with-asr', action='store_true', help='also compute the composed Layer 1+2 row (slow: runs Whisper again)')
    p2.add_argument('--model', default='tiny', help="faster-whisper model size: tiny/base/small")
    p2.add_argument('--out', default=None)
    p2.set_defaults(func=cmd_layer2)

    pc = sub.add_parser('check-align', help='diagnostic: MFA word starts vs GRID .align word starts')
    pc.add_argument('grid_corpus')
    pc.add_argument('mfa_out')
    pc.add_argument('--speakers', required=True)
    pc.add_argument('--limit', type=int, default=50)
    pc.set_defaults(func=cmd_check_align)

    pr = sub.add_parser('report')
    pr.add_argument('layer1')
    pr.add_argument('layer2')
    pr.add_argument('--out', default=None)
    pr.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
