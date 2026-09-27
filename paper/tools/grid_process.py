#!/usr/bin/env python3
"""
GRID Corpus → TranscriptionSync Training Dataset
Tek dosya, yalnızca numpy gerekli.

Üretilen veri: makale-2 (öğrenilmiş model) eğitim seti.
Makale-1'in V-D validation bölümü için DEĞİL — orada ham .align + MFA kullanılır.

Düzeltmeler (önceki sürüme göre):
  1. Zaman birimi: GRID .align değerleri 25 kHz örnek birimidir (saniye = değer/25000).
     Birim otomatik tespit edilir; 1000x hata sınıfı tamamen engellenir.
  2. Harf token'ları harf ADIYLA fonemlenir (b → B IY "bee", a → EY).
  3. Gaussian smoothing simetrik (offline etiket — nedensel filtre faz gecikmesi işlerdi).
  4. Çıkış gerçek 14 ARKit mouth kanalıdır (jawOpen, mouthClose, ...) — viseme uzayından
     VISEME_TO_ARKIT ile dönüştürülür. Viseme indeksi de saklanır.
  5. Split: s1–s28 train, s29–s34 val (makale V-D ile uyumlu, speaker-disjoint).

Kullanım:
  python grid_process.py --corpus ~/grid_corpus --output ~/grid_dataset
"""

import json
import argparse
from pathlib import Path
import numpy as np

# ── Sabitler ─────────────────────────────────────────────────────────────────
OUTPUT_FPS = 60
FRAME_MS   = 1000.0 / OUTPUT_FPS          # ~16.67 ms
VAL_SPEAKER_MIN = 29                       # s29–s34 → val (speaker-disjoint)
EXPECTED_UTT_SEC = 3.0                     # GRID cümleleri ~3 s

# 14 viseme sınıfı (Oculus-stili ara temsil)
VISEME_CLASSES = ["sil","aa","E","I","O","U","PP","FF","TH","DD","kk","CH","SS","nn"]
N_VISEMES = 14

# 14 ARKit mouth kanalı (Atlas avatar hattıyla birebir)
ARKIT_CHANNELS = [
    "jawOpen","mouthClose","mouthFunnel","mouthPucker",
    "mouthStretchLeft","mouthStretchRight",
    "mouthUpperUpLeft","mouthUpperUpRight",
    "mouthLowerDownLeft","mouthLowerDownRight",
    "mouthShrugUpper","mouthRollLower",
    "mouthDimpleLeft","mouthDimpleRight",
]
N_ARKIT = 14

# ── İngilizce fonem → viseme indeksi ─────────────────────────────────────────
EN_PHONEME_TO_VISEME = {
    "SIL":0,"SP":0,"":0,
    "AA":1,"AE":1,"AH":1,"AW":1,"AY":1,
    "EH":2,"EY":2,"ER":2,
    "IH":3,"IY":3,
    "AO":4,"OW":4,"OY":4,
    "UH":5,"UW":5,
    "P":6,"B":6,"M":6,
    "F":7,"V":7,
    "TH":8,"DH":8,
    "D":9,"T":9,"N":9,"L":9,
    "K":10,"G":10,"NG":10,"HH":10,
    "CH":11,"JH":11,"Y":11,
    "S":12,"Z":12,"SH":12,"ZH":12,
    "R":13,"W":13,
}

# ── Viseme sınıfı → ARKit hedefleri (Atlas main.py viseme tablosundan) ────────
def _arkit(**kw):
    v = np.zeros(N_ARKIT, dtype=np.float32)
    idx = {name: i for i, name in enumerate(ARKIT_CHANNELS)}
    for k, val in kw.items():
        v[idx[k]] = val
    return v

VISEME_TO_ARKIT = np.stack([
    _arkit(mouthClose=0.15),                                                      # sil
    _arkit(jawOpen=0.60, mouthLowerDownLeft=0.35, mouthLowerDownRight=0.35,
           mouthUpperUpLeft=0.18, mouthUpperUpRight=0.18),                        # aa
    _arkit(jawOpen=0.32, mouthStretchLeft=0.42, mouthStretchRight=0.42,
           mouthLowerDownLeft=0.18, mouthLowerDownRight=0.18),                    # E
    _arkit(jawOpen=0.18, mouthStretchLeft=0.52, mouthStretchRight=0.52),          # I
    _arkit(jawOpen=0.42, mouthFunnel=0.38, mouthPucker=0.18,
           mouthLowerDownLeft=0.22, mouthLowerDownRight=0.22),                    # O
    _arkit(jawOpen=0.18, mouthPucker=0.58, mouthFunnel=0.42),                     # U
    _arkit(jawOpen=0.02, mouthClose=0.80, mouthShrugUpper=0.15),                  # PP
    _arkit(jawOpen=0.07, mouthUpperUpLeft=0.53, mouthUpperUpRight=0.53,
           mouthLowerDownLeft=0.12, mouthLowerDownRight=0.12),                    # FF
    _arkit(jawOpen=0.15, mouthLowerDownLeft=0.08, mouthLowerDownRight=0.08),      # TH
    _arkit(jawOpen=0.12, mouthStretchLeft=0.11, mouthStretchRight=0.11),          # DD
    _arkit(jawOpen=0.21),                                                         # kk
    _arkit(jawOpen=0.12, mouthStretchLeft=0.17, mouthStretchRight=0.17,
           mouthPucker=0.10),                                                     # CH
    _arkit(jawOpen=0.08, mouthStretchLeft=0.25, mouthStretchRight=0.25),          # SS
    _arkit(jawOpen=0.17, mouthPucker=0.30, mouthFunnel=0.25),                     # nn
])

# ── Fonem süre ağırlıkları (λ öncelikleri) ────────────────────────────────────
DURATION_WEIGHTS = {
    "AA":1.4,"AE":1.3,"AH":1.0,"AO":1.3,"AW":1.5,"AY":1.5,
    "EH":1.2,"ER":1.3,"EY":1.4,"IH":0.9,"IY":1.3,
    "OW":1.4,"OY":1.4,"UH":1.0,"UW":1.3,
    "P":0.7,"B":0.7,"T":0.7,"D":0.7,"K":0.7,"G":0.7,
    "M":0.9,"N":0.9,"NG":0.9,
    "F":0.9,"V":0.9,"TH":1.0,"DH":1.0,
    "S":1.0,"Z":1.0,"SH":1.1,"ZH":1.1,"HH":0.8,
    "CH":1.0,"JH":1.0,"L":0.9,"R":0.9,"W":0.8,"Y":0.8,
    "SIL":0.5,
}

# ── G2P: GRID söz dağarcığı ───────────────────────────────────────────────────
# GRID grameri: komut(bin/lay/place/set) renk(blue/green/red/white)
# edat(at/by/in/with) HARF(a–z, w hariç) RAKAM(zero–nine) zarf(again/now/please/soon)
COMMON_WORDS = {
    "bin":["B","IH","N"], "lay":["L","EY"], "place":["P","L","EY","S"],
    "set":["S","EH","T"],
    "blue":["B","L","UW"], "green":["G","R","IY","N"], "red":["R","EH","D"],
    "white":["W","AY","T"],
    "at":["AE","T"], "by":["B","AY"], "in":["IH","N"], "with":["W","IH","DH"],
    "zero":["Z","IH","R","OW"], "one":["W","AH","N"], "two":["T","UW"],
    "three":["TH","R","IY"], "four":["F","AO","R"], "five":["F","AY","V"],
    "six":["S","IH","K","S"], "seven":["S","EH","V","AH","N"],
    "eight":["EY","T"], "nine":["N","AY","N"],
    "again":["AH","G","EH","N"], "now":["N","AW"],
    "please":["P","L","IY","Z"], "soon":["S","UW","N"],
    "the":["DH","AH"], "a":["EY"], "and":["AE","N","D"],
}

# GRID'de tek harf token'ları harf ADIYLA okunur: "b" → "bee" → B IY
LETTER_NAMES = {
    "a":["EY"], "b":["B","IY"], "c":["S","IY"], "d":["D","IY"], "e":["IY"],
    "f":["EH","F"], "g":["JH","IY"], "h":["EY","CH"], "i":["AY"],
    "j":["JH","EY"], "k":["K","EY"], "l":["EH","L"], "m":["EH","M"],
    "n":["EH","N"], "o":["OW"], "p":["P","IY"], "q":["K","Y","UW"],
    "r":["AA","R"], "s":["EH","S"], "t":["T","IY"], "u":["Y","UW"],
    "v":["V","IY"], "x":["EH","K","S"], "y":["W","AY"], "z":["Z","IY"],
    # GRID'de 'w' yoktur; bazı dağıtımlarda 'zed' geçer:
    "zed":["Z","EH","D"],
}

# Sözlük dışı kelimeler için kaba fallback
LETTER_MAP = {
    "a":"AE","b":"B","c":"K","d":"D","e":"EH","f":"F","g":"G","h":"HH",
    "i":"IH","j":"JH","k":"K","l":"L","m":"M","n":"N","o":"AO","p":"P",
    "q":"K","r":"R","s":"S","t":"T","u":"UH","v":"V","w":"W","x":"K",
    "y":"Y","z":"Z",
}

def g2p(word: str):
    w = word.lower().strip(".,!?")
    if w in COMMON_WORDS:
        return COMMON_WORDS[w]
    if len(w) == 1 and w in LETTER_NAMES:
        return LETTER_NAMES[w]
    if w in LETTER_NAMES:
        return LETTER_NAMES[w]
    phones, i = [], 0
    while i < len(w):
        if i + 1 < len(w):
            di = w[i:i+2]
            if di == "th": phones.append("TH"); i += 2; continue
            if di == "sh": phones.append("SH"); i += 2; continue
            if di == "ch": phones.append("CH"); i += 2; continue
            if di == "ng": phones.append("NG"); i += 2; continue
            if di in ("ee","ea"): phones.append("IY"); i += 2; continue
            if di in ("oo","ou"): phones.append("UW"); i += 2; continue
        if w[i].isalpha():
            phones.append(LETTER_MAP.get(w[i], "SIL"))
        i += 1
    return phones or ["SIL"]

# ── Süre dağıtımı (Eq. 4: word-bounded duration estimation) ──────────────────
def estimate_durations(phones, word_ms):
    weights = [DURATION_WEIGHTS.get(p, 0.9) for p in phones]
    total = sum(weights)
    return [(wgt / total) * word_ms for wgt in weights]

# ── Frame sentezi ─────────────────────────────────────────────────────────────
def gaussian_smooth_symmetric(weights: np.ndarray, sigma_ms: float) -> np.ndarray:
    """Simetrik Gaussian — offline etiket üretimi; faz gecikmesi yoktur."""
    sigma_f = sigma_ms / FRAME_MS
    radius  = max(1, int(3 * sigma_f))
    k = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (k / sigma_f) ** 2)
    kernel /= kernel.sum()
    out = np.empty_like(weights)
    for c in range(weights.shape[1]):
        out[:, c] = np.convolve(weights[:, c], kernel, mode="same")
    return out

def words_to_frames(words, sigma_ms: float):
    """[(word, start_s, end_s)] → 60 fps ARKit frame listesi."""
    segments = []
    for word, start_s, end_s in words:
        dur_ms = (end_s - start_s) * 1000.0
        phones = g2p(word)
        durs   = estimate_durations(phones, dur_ms)
        t = start_s * 1000.0
        for ph, d in zip(phones, durs):
            vi = EN_PHONEME_TO_VISEME.get(ph, 0)
            segments.append((ph, vi, t, t + d))
            t += d
    if not segments:
        return []

    end_ms = segments[-1][3] + 200.0
    n_frames = int(end_ms / FRAME_MS) + 1
    viseme_act = np.zeros((n_frames, N_VISEMES), dtype=np.float32)
    viseme_idx = np.zeros(n_frames, dtype=np.int32)
    ANTICIPATION = 30.0  # ms — artikülatör hazırlık

    seg_i = 0
    for f in range(n_frames):
        t_look = f * FRAME_MS + ANTICIPATION
        # segment ilerlet (segmentler zaman-sıralı)
        while seg_i + 1 < len(segments) and t_look >= segments[seg_i][3]:
            seg_i += 1
        ph, vi, s, e = segments[seg_i]
        if s <= t_look < e:
            prog = (t_look - s) / max(e - s, 1.0)
            if prog > 0.70 and seg_i + 1 < len(segments):
                alpha = (prog - 0.70) / 0.30      # sonraki vizeme crossfade
                nxt = segments[seg_i + 1][1]
                viseme_act[f, vi]  = 1.0 - alpha
                viseme_act[f, nxt] = alpha
            else:
                viseme_act[f, vi] = 1.0
            viseme_idx[f] = vi
        else:
            viseme_act[f, 0] = 1.0                # kelime arası boşluk → sil
            viseme_idx[f] = 0

    viseme_act = gaussian_smooth_symmetric(viseme_act, sigma_ms)
    arkit = np.clip(viseme_act @ VISEME_TO_ARKIT, 0.0, 1.0)

    return [
        {
            "t":  round(f * FRAME_MS, 2),
            "a":  [round(float(x), 3) for x in arkit[f]],
            "vi": int(viseme_idx[f]),
        }
        for f in range(n_frames)
    ]

# ── GRID parser ───────────────────────────────────────────────────────────────
def detect_time_divisor(align_dir: Path) -> float:
    """Align birimi otomatik tespiti: medyan cümle sonunu ~3 s'ye getiren bölen.
    Standart GRID: 25 kHz örnek birimi → 25000. Bazı dağıtımlar ms ya da
    video-frame kullanır; hepsi yakalanır."""
    ends = []
    for ap in sorted(align_dir.glob("*.align"))[:20]:
        try:
            last = 0
            for line in ap.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    last = max(last, float(parts[1]))
            if last > 0:
                ends.append(last)
        except Exception:
            continue
    if not ends:
        return 25000.0
    med = float(np.median(ends))
    candidates = [25000.0, 1000.0, 100.0, 25.0, 1.0]
    return min(candidates, key=lambda d: abs(med / d - EXPECTED_UTT_SEC))

def parse_align(path: Path, divisor: float):
    words = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        sf, ef, word = float(parts[0]), float(parts[1]), parts[2]
        if word.lower() in ("sil", "sp"):
            continue
        words.append((word, sf / divisor, ef / divisor))
    return words

# ── Ana akış ──────────────────────────────────────────────────────────────────
def process_grid(corpus_root: str, output_dir: str, sigma_ms: float):
    corpus = Path(corpus_root).expanduser()
    out    = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "grid_all.jsonl"

    total, errors = 0, 0
    durations = []
    divisor_logged = None

    print(f"Corpus : {corpus}")
    print(f"Output : {out_file}")
    print(f"Sigma  : {sigma_ms} ms (simetrik)   FPS: {OUTPUT_FPS}")
    print(f"Split  : s1–s{VAL_SPEAKER_MIN-1} train | s{VAL_SPEAKER_MIN}–s34 val\n")

    with open(out_file, "w", encoding="utf-8") as fout:
        for spk_num in range(1, 35):
            spk_dir   = corpus / f"s{spk_num}"
            align_dir = spk_dir / "align"
            wav_dir   = spk_dir / f"s{spk_num}"
            if not align_dir.exists():
                continue

            divisor = detect_time_divisor(align_dir)
            if divisor_logged != divisor:
                divisor_logged = divisor
                unit = {25000.0: "25 kHz örnek", 1000.0: "ms",
                        100.0: "10 ms", 25.0: "video frame", 1.0: "saniye"}.get(divisor, "?")
                print(f"  [birim] .align böleni = {divisor:g}  ({unit})")

            spk_count = 0
            for ap in sorted(align_dir.glob("*.align")):
                try:
                    words = parse_align(ap, divisor)
                    if not words:
                        continue
                    dur_ms = words[-1][2] * 1000.0
                    if not (500.0 < dur_ms < 15000.0):
                        errors += 1
                        continue  # birim/parse anomalisi — sessizce atlama yok, sayılır

                    frames = words_to_frames(words, sigma_ms)
                    if not frames:
                        continue

                    wav_path = wav_dir / f"{ap.stem}.wav"
                    sample = {
                        "id":    f"grid_s{spk_num}_{ap.stem}",
                        "lang":  "en",
                        "spk":   f"s{spk_num}",
                        "wav":   str(wav_path.relative_to(corpus)) if wav_path.exists() else None,
                        "words": [{"w": w, "s": round(s, 4), "e": round(e, 4)}
                                  for w, s, e in words],
                        "fps":      OUTPUT_FPS,
                        "channels": ARKIT_CHANNELS,
                        "frames":   frames,
                        "dur_ms":   round(dur_ms, 1),
                        "split":    "val" if spk_num >= VAL_SPEAKER_MIN else "train",
                    }
                    fout.write(json.dumps(sample) + "\n")
                    durations.append(dur_ms)
                    total += 1
                    spk_count += 1
                except Exception:
                    errors += 1

            print(f"  s{spk_num:2d}: {spk_count:4d} utterance")

    print()
    if durations:
        avg = float(np.mean(durations))
        print(f"✅ Toplam    : {total:,} sample")
        print(f"⚠️  Hata/atlanan: {errors}")
        print(f"📊 Ort. süre : {avg:.0f} ms  (beklenen ~3000 ms)")
        print(f"⏱️  Top. süre : {sum(durations)/1000/3600:.2f} saat")
        print(f"📁 Dosya     : {out_file}")
        if not (1500 < avg < 8000):
            print("\n❌ UYARI: ortalama süre beklenen aralıkta değil — birim tespiti")
            print("   şüpheli. Bir .align dosyasını wav süresiyle elle karşılaştırın.")
        else:
            print("\n🚀 Eğitime hazır.")
    else:
        print("❌ Hiç sample üretilemedi — corpus yolunu ve yapıyı kontrol edin.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GRID → TranscriptionSync training dataset")
    p.add_argument("--corpus", default="~/grid_corpus")
    p.add_argument("--output", default="~/grid_dataset")
    p.add_argument("--sigma",  type=float, default=40.0,
                   help="Gaussian smoothing sigma (ms), simetrik. Varsayılan: 40")
    args = p.parse_args()
    process_grid(args.corpus, args.output, args.sigma)
