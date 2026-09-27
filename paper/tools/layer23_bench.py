"""
Standalone implementation + latency/CPU benchmark of TranscriptionSync Layer 2
(G2P + word-bounded phoneme duration estimation, Eq. 4-5) and Layer 3 (kernel
co-articulated 14-channel ARKit viseme synthesis, Eq. 6-7), as specified in
Section IV-C/D of the ATLAS IEEE Access draft.

This module does NOT depend on Layer 1 (streaming ASR) — it consumes
finalized (text, t_start, t_end) word triples, exactly the interface Layer 1
would hand off via the playout buffer (Eq. 3). Run directly to reproduce the
Layer-2/3 rows of Table 6 (latency budget) on the local machine:

    python3 layer23_bench.py
"""
import random
import re
import time

import cmudict
import numpy as np
import psutil

# ── 14 ARKit mouth channels ────────────────────────────────────────────────
CHANNELS = (
    'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker',
    'MouthStretchLeft', 'MouthStretchRight',
    'MouthUpperUpLeft', 'MouthUpperUpRight',
    'MouthLowerDownLeft', 'MouthLowerDownRight',
    'MouthShrugUpper', 'MouthRollLower',
    'MouthDimpleLeft', 'MouthDimpleRight',
)
_Z = {c: 0.0 for c in CHANNELS}


def _v(**kw):
    d = dict(_Z)
    d.update(kw)
    return d


# ── Eq. (4) articulatory-class duration priors lambda ─────────────────────
DUR_PRIOR = {
    'vowel': 1.6, 'diphthong': 2.1, 'fricative': 1.0, 'affricate': 0.9,
    'nasal': 0.9, 'plosive': 0.7, 'liquid': 0.8, 'glide': 0.75,
}

# ── English (ARPAbet, stress-stripped) -> (duration class, viseme target) ──
_OPEN_VOWEL = _v(JawOpen=0.60, MouthLowerDownLeft=0.35, MouthLowerDownRight=0.35,
                  MouthUpperUpLeft=0.18, MouthUpperUpRight=0.18)
_SPREAD_VOWEL = _v(JawOpen=0.28, MouthStretchLeft=0.46, MouthStretchRight=0.46,
                    MouthLowerDownLeft=0.16, MouthLowerDownRight=0.16)
_ROUND_VOWEL = _v(JawOpen=0.30, MouthFunnel=0.40, MouthPucker=0.38,
                   MouthLowerDownLeft=0.18, MouthLowerDownRight=0.18)
_MID_VOWEL = _v(JawOpen=0.22, MouthStretchLeft=0.15, MouthStretchRight=0.15)
_BILABIAL = _v(JawOpen=0.02, MouthClose=0.85, MouthShrugUpper=0.12)
_LABIODENTAL = _v(JawOpen=0.06, MouthUpperUpLeft=0.55, MouthUpperUpRight=0.55,
                   MouthLowerDownLeft=0.10, MouthLowerDownRight=0.10)
_SIBILANT = _v(JawOpen=0.08, MouthStretchLeft=0.25, MouthStretchRight=0.25)
_AFFRICATE = _v(JawOpen=0.10, MouthFunnel=0.20, MouthStretchLeft=0.18, MouthStretchRight=0.18)
_DENTAL_FRIC = _v(JawOpen=0.10, MouthUpperUpLeft=0.30, MouthUpperUpRight=0.30)
_NASAL = _v(JawOpen=0.08, MouthShrugUpper=0.05)
_LIQUID = _v(JawOpen=0.18, MouthLowerDownLeft=0.12, MouthLowerDownRight=0.12)
_GLIDE_W = _v(JawOpen=0.14, MouthPucker=0.50, MouthFunnel=0.36)
_GLIDE_Y = _v(JawOpen=0.14, MouthStretchLeft=0.20, MouthStretchRight=0.20)
_GLOTTAL = _v(JawOpen=0.26)
_VELAR = _v(JawOpen=0.20)
_ALVEOLAR = _v(JawOpen=0.12, MouthStretchLeft=0.10, MouthStretchRight=0.10)
_DEFAULT_CONS = _v(JawOpen=0.12, MouthShrugUpper=0.08)
_GAP = _v(JawOpen=0.04, MouthClose=0.28)
BILABIAL_PHONES_EN = {'M', 'B', 'P'}

ARPABET_TABLE = {
    # monophthong vowels
    'AA': ('vowel', _OPEN_VOWEL), 'AE': ('vowel', _OPEN_VOWEL),
    'AH': ('vowel', _MID_VOWEL), 'AO': ('vowel', _ROUND_VOWEL),
    'AX': ('vowel', _MID_VOWEL), 'EH': ('vowel', _SPREAD_VOWEL),
    'ER': ('vowel', _MID_VOWEL), 'IH': ('vowel', _SPREAD_VOWEL),
    'IX': ('vowel', _SPREAD_VOWEL), 'IY': ('vowel', _SPREAD_VOWEL),
    'UH': ('vowel', _ROUND_VOWEL), 'UW': ('vowel', _ROUND_VOWEL),
    'UX': ('vowel', _ROUND_VOWEL),
    # diphthongs
    'AW': ('diphthong', _OPEN_VOWEL), 'AY': ('diphthong', _OPEN_VOWEL),
    'EY': ('diphthong', _SPREAD_VOWEL), 'OW': ('diphthong', _ROUND_VOWEL),
    'OY': ('diphthong', _ROUND_VOWEL),
    # plosives
    'B': ('plosive', _BILABIAL), 'P': ('plosive', _BILABIAL),
    'D': ('plosive', _ALVEOLAR), 'T': ('plosive', _ALVEOLAR),
    'G': ('plosive', _VELAR), 'K': ('plosive', _VELAR),
    # fricatives
    'F': ('fricative', _LABIODENTAL), 'V': ('fricative', _LABIODENTAL),
    'TH': ('fricative', _DENTAL_FRIC), 'DH': ('fricative', _DENTAL_FRIC),
    'S': ('fricative', _SIBILANT), 'Z': ('fricative', _SIBILANT),
    'SH': ('fricative', _SIBILANT), 'ZH': ('fricative', _SIBILANT),
    'HH': ('fricative', _GLOTTAL),
    # affricates
    'CH': ('affricate', _AFFRICATE), 'JH': ('affricate', _AFFRICATE),
    # nasals
    'M': ('nasal', _BILABIAL), 'N': ('nasal', _NASAL), 'NG': ('nasal', _NASAL),
    # liquids
    'L': ('liquid', _LIQUID), 'R': ('liquid', _LIQUID),
    # glides
    'W': ('glide', _GLIDE_W), 'Y': ('glide', _GLIDE_Y),
}

# ── Turkish (near 1:1 grapheme-phoneme) -> (duration class, viseme target) ─
TURKISH_TABLE = {
    'a': ('vowel', _OPEN_VOWEL), 'e': ('vowel', _SPREAD_VOWEL),
    'ı': ('vowel', _MID_VOWEL), 'i': ('vowel', _SPREAD_VOWEL),
    'o': ('vowel', _ROUND_VOWEL), 'ö': ('vowel', _ROUND_VOWEL),
    'u': ('vowel', _ROUND_VOWEL), 'ü': ('vowel', _ROUND_VOWEL),
    'b': ('plosive', _BILABIAL), 'p': ('plosive', _BILABIAL),
    'd': ('plosive', _ALVEOLAR), 't': ('plosive', _ALVEOLAR),
    'g': ('plosive', _VELAR), 'k': ('plosive', _VELAR),
    'f': ('fricative', _LABIODENTAL), 'v': ('fricative', _LABIODENTAL),
    's': ('fricative', _SIBILANT), 'z': ('fricative', _SIBILANT),
    'ş': ('fricative', _SIBILANT), 'j': ('fricative', _SIBILANT),
    'h': ('fricative', _GLOTTAL),
    'c': ('affricate', _AFFRICATE), 'ç': ('affricate', _AFFRICATE),
    'm': ('nasal', _BILABIAL), 'n': ('nasal', _NASAL),
    'l': ('liquid', _LIQUID), 'r': ('liquid', _LIQUID),
    'y': ('glide', _GLIDE_Y), 'ğ': ('glide', _GLIDE_W),
}
BILABIAL_PHONES_TR = {'b', 'p', 'm'}

CMU = cmudict.dict()
_WORD_RE = re.compile(r"[a-zA-Z']+")


def g2p_en(word: str) -> list[str]:
    """English G2P via CMUdict; falls back to a per-letter approximation for OOV words."""
    key = word.lower()
    entries = CMU.get(key)
    if entries:
        return [re.sub(r'\d', '', p) for p in entries[0]]
    # OOV fallback: map each letter to a plausible ARPAbet phone
    fallback = {'a': 'AA', 'e': 'EH', 'i': 'IH', 'o': 'OW', 'u': 'UH',
                'y': 'Y', 'q': 'K', 'x': 'K', 'w': 'W'}
    return [fallback.get(c, c.upper()) for c in key if c.isalpha()]


def g2p_tr(word: str) -> list[str]:
    """Turkish G2P: rule-based, near 1:1 grapheme-to-phoneme (shallow orthography)."""
    return [c for c in word.lower() if c in TURKISH_TABLE]


# ── Layer 2: word-bounded phoneme duration estimation (Eq. 4-5) ───────────
def layer2_durations(phonemes: list[str], t_start: float, t_end: float, table: dict):
    """Given phonemes and word interval [t_start, t_end), return (durations, midpoints)."""
    T_i = t_end - t_start
    if not phonemes:
        return [], []
    lambdas = [DUR_PRIOR[table[p][0]] for p in phonemes]
    total = sum(lambdas)
    durations = [T_i * lam / total for lam in lambdas]
    midpoints, acc = [], 0.0
    for d in durations:
        midpoints.append(t_start + acc + d / 2.0)
        acc += d
    return durations, midpoints


# ── Layer 3: kernel co-articulated synthesis (Eq. 6-7) ─────────────────────
SIGMA = 0.030  # 30 ms, per Section IV-D design value
SUPPORT = 3 * SIGMA
EPS = 1e-6


def layer3_render(timed_phonemes, table: dict, bilabial_set, fps: int = 60):
    """timed_phonemes: list of (phoneme, midpoint_s, viseme_target_dict, is_bilabial).
    Renders 60 fps frames over the utterance span; returns list of per-channel dicts
    and the per-frame wall-clock synthesis times (s)."""
    if not timed_phonemes:
        return [], []
    t0 = timed_phonemes[0][1] - SUPPORT
    t1 = timed_phonemes[-1][1] + SUPPORT
    n_frames = max(1, int((t1 - t0) * fps))
    frames, frame_times = [], []
    for f in range(n_frames):
        t = t0 + f / fps
        tic = time.perf_counter()
        # restrict to phonemes within +/- 3 sigma (J_local <= 5 by construction)
        local = [(p, tau, V, bl) for (p, tau, V, bl) in timed_phonemes if abs(t - tau) <= SUPPORT]
        if not local:
            frame_times.append(time.perf_counter() - tic)
            frames.append(dict(_GAP))
            continue
        weights = np.array([np.exp(-((t - tau) ** 2) / (2 * SIGMA ** 2)) for (_, tau, _, _) in local])
        wsum = max(EPS, float(weights.sum()))
        frame = {}
        for ch in CHANNELS:
            if ch == 'MouthClose':
                continue
            vals = np.array([V[ch] for (_, _, V, _) in local])
            frame[ch] = float((weights * vals).sum() / wsum)
        # Eq. (7): closure-dominance rule for MouthClose, winner-take-all over bilabials
        bilabial_w = [(w, V['MouthClose']) for (_, _, V, bl), w in zip(local, weights) if bl]
        if bilabial_w:
            frame['MouthClose'] = max(w * vc for w, vc in bilabial_w)
            frame['JawOpen'] *= (1.0 - frame['MouthClose'])  # reciprocal suppression
        else:
            vals = np.array([V['MouthClose'] for (_, _, V, _) in local])
            frame['MouthClose'] = float((weights * vals).sum() / wsum)
        frame_times.append(time.perf_counter() - tic)
        frames.append(frame)
    return frames, frame_times


# ── Test utterances ─────────────────────────────────────────────────────
EN_SENTENCES = [
    "the sun rose over the quiet harbor this morning",
    "she quickly opened the door and stepped outside",
    "artificial intelligence is changing how we work",
    "please remember to bring your charger tomorrow",
    "the train departs from platform six in ten minutes",
]
TR_SENTENCES = [
    "güneş bu sabah limanın üzerinde yükseldi",
    "kapıyı hızla açtı ve dışarı çıktı",
    "yapay zeka çalışma şeklimizi değiştiriyor",
    "lütfen şarj cihazını yarın getirmeyi unutma",
    "tren on dakika içinde altıncı platformdan kalkıyor",
]


def synth_word_timestamps(words, rng):
    """Synthetic (text, t_start, t_end) triples: ~80ms/phoneme + inter-word gaps,
    matching the order of magnitude of natural connected speech."""
    t = 0.0
    out = []
    for w in words:
        n_ph = max(1, len(re.sub(r'[^a-zA-Zığüşöç]', '', w.lower())) - 1)
        dur = n_ph * rng.uniform(0.060, 0.100)
        out.append((w, t, t + dur))
        t += dur + rng.uniform(0.04, 0.14)  # inter-word gap
    return out


def run_benchmark(n_reps: int = 30):
    rng = random.Random(42)
    layer2_times_ms = []  # per word: G2P + duration estimation (L_g2p)
    layer3_times_ms = []  # per frame: kernel synthesis (L_syn)

    for _ in range(n_reps):
        for lang, sentences, table, bilset, g2p in (
            ('en', EN_SENTENCES, ARPABET_TABLE, BILABIAL_PHONES_EN, g2p_en),
            ('tr', TR_SENTENCES, TURKISH_TABLE, BILABIAL_PHONES_TR, g2p_tr),
        ):
            sentence = rng.choice(sentences)
            words = sentence.split()
            triples = synth_word_timestamps(words, rng)
            timed_phonemes = []
            for word, t_s, t_e in triples:
                tic = time.perf_counter()
                phonemes = g2p(word)
                _, midpoints = layer2_durations(phonemes, t_s, t_e, table)
                layer2_times_ms.append((time.perf_counter() - tic) * 1000.0)
                for p, tau in zip(phonemes, midpoints):
                    cls, V = table[p]
                    is_bl = p in bilset
                    timed_phonemes.append((p, tau, V, is_bl))
                # inter-word gap pose at the word boundary
            timed_phonemes.sort(key=lambda x: x[1])
            _, frame_times = layer3_render(timed_phonemes, table, bilset)
            layer3_times_ms.extend(ft * 1000.0 for ft in frame_times)

    return np.array(layer2_times_ms), np.array(layer3_times_ms)


def duty_cycle_cpu_pct(l2_ms, l3_ms, words_per_s: float = 2.5, fps: int = 60):
    """CPU occupancy of one core under the *real* duty cycle: Layer 3 runs once
    per rendered frame (period 1000/fps ms); Layer 2 runs once per finalized
    word (period 1000/words_per_s ms, ~2.5 words/s for connected speech).
    Busy-looping the benchmark itself (no sleeps) would peg one core at ~100%
    and is not representative — occupancy is cost / period."""
    frame_period_ms = 1000.0 / fps
    word_period_ms = 1000.0 / words_per_s
    l3_occ = float(np.median(l3_ms)) / frame_period_ms
    l2_occ = float(np.median(l2_ms)) / word_period_ms
    return (l2_occ + l3_occ) * 100.0


def pct(a, q):
    return float(np.percentile(a, q))


if __name__ == '__main__':
    import platform
    l2, l3 = run_benchmark(n_reps=30)
    cpu_pct = duty_cycle_cpu_pct(l2, l3)

    print(f"Platform: {platform.processor() or platform.machine()}, "
          f"{psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical cores")
    print(f"n words processed (Layer 2): {len(l2)}")
    print(f"n frames processed (Layer 3): {len(l3)}")
    print()
    print("| Stage | Definition | Median | P95 |")
    print("|---|---|---|---|")
    print(f"| L_g2p (G2P + Eq.4-5 duration est.), per word (ms) | Layer 2 | "
          f"{pct(l2, 50):.4f} | {pct(l2, 95):.4f} |")
    print(f"| L_syn (Eq.6-7 kernel synthesis), per frame (ms) | Layer 3 | "
          f"{pct(l3, 50):.4f} | {pct(l3, 95):.4f} |")
    print()
    print(f"Layer-2/3 CPU load (one core, duty-cycle estimate @ 60fps, 2.5 words/s): {cpu_pct:.4f}%")
    print()
    print(f"L_g2p + L_syn (P95, ms): {pct(l2, 95) + pct(l3, 95):.4f}")
