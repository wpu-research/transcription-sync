#!/usr/bin/env python3
"""
GRID videoları → MediaPipe FaceLandmarker → 14 ARKit kanal GERÇEK ground-truth
Çıktı, grid_process.py ile aynı JSONL formatındadır → grid_train.py / grid_eval.py
HİÇ DEĞİŞMEDEN bu GT ile çalışır.

GEREKSİNİM (Windows):
  pip install mediapipe opencv-python
  grid_process.py ile AYNI klasörde durmalı (align parser'ı oradan alır).
  face_landmarker.task modeli yoksa otomatik indirilir (~4 MB).

Kullanım:
  python grid_video_gt.py --video C:\\...\\grid_video --corpus C:\\...\\grid_corpus --out C:\\...\\grid_gt --merge

Notlar:
  - Video 25 fps → blendshape'ler 60 fps'e lineer interpolasyonla örneklenir.
  - Konuşmacı başına ayrı sN.jsonl yazılır; yarıda kesilirse aynı komut kaldığı
    yerden sürer (işlenmiş id'ler atlanır). --merge sonda hepsini birleştirir.
  - Yüz tespiti %80'in altında kalan cümleler atılır (raporlanır).
  - s21'in videosu yoktur; --speakers verilmezse otomatik atlanır.
"""

import json
import time
import argparse
from pathlib import Path
import numpy as np

VIDEO_FPS_DEFAULT = 25.0
OUTPUT_FPS = 60
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# grid_process.py / grid_train.py ile birebir aynı kanal sırası
ARKIT_CHANNELS = [
    "jawOpen","mouthClose","mouthFunnel","mouthPucker",
    "mouthStretchLeft","mouthStretchRight",
    "mouthUpperUpLeft","mouthUpperUpRight",
    "mouthLowerDownLeft","mouthLowerDownRight",
    "mouthShrugUpper","mouthRollLower",
    "mouthDimpleLeft","mouthDimpleRight",
]
VIDEO_EXTS = (".mpg", ".mpeg", ".mp4", ".avi", ".mov", ".m4v")
NO_VIDEO_SPEAKERS = {21}
VAL_SPEAKER_MIN = 29


# ── Model dosyası ─────────────────────────────────────────────────────────────
def ensure_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    print(f"⬇ face_landmarker.task indiriliyor → {model_path}")
    from urllib.request import urlopen
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(MODEL_URL, timeout=60) as r:
        model_path.write_bytes(r.read())
    return model_path


# ── Video birimi keşfi: dosya (.mpg/.mp4) ya da jpg-frame klasörü ─────────────
def find_units(speaker_dir: Path):
    """[(stem, 'video', path)] veya [(stem, 'frames', dir)] listesi döner."""
    units = {}
    for p in speaker_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            units.setdefault(p.stem, ("video", p))
    if not units:  # jpg-frame dağıtımı: her cümle bir klasör dolusu jpg
        for d in sorted(speaker_dir.rglob("*")):
            if d.is_dir():
                jpgs = sorted(d.glob("*.jpg")) or sorted(d.glob("*.jpeg"))
                if len(jpgs) >= 10:
                    units.setdefault(d.name, ("frames", d))
    return [(stem, kind, path) for stem, (kind, path) in sorted(units.items())]


def iter_frames(kind: str, path: Path):
    """(rgb_frame, fps) üretir."""
    import cv2
    if kind == "video":
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS_DEFAULT
        if not (5.0 < fps < 120.0):
            fps = VIDEO_FPS_DEFAULT
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fps
        cap.release()
    else:  # jpg frame klasörü — GRID frame dağıtımları 25 fps'tir
        for jp in sorted(path.glob("*.jp*g")):
            bgr = cv2.imread(str(jp))
            if bgr is not None:
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), VIDEO_FPS_DEFAULT


# ── Blendshape çıkarımı ───────────────────────────────────────────────────────
class BlendshapeExtractor:
    """Konuşmacı başına tek landmarker; timestamp global monoton tutulur."""

    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self._ts_ms = 0  # global monoton timestamp

    def extract(self, kind: str, path: Path):
        """(B25, fps, detect_rate): B25 = (T, 14) float32, tespit edilemeyen
        frame'lerde son bilinen değer taşınır."""
        rows, misses, fps = [], 0, VIDEO_FPS_DEFAULT
        last = np.zeros(len(ARKIT_CHANNELS), dtype=np.float32)
        idx = {n: i for i, n in enumerate(ARKIT_CHANNELS)}
        for rgb, fps in iter_frames(kind, path):
            self._ts_ms += int(1000.0 / fps)
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.lm.detect_for_video(img, self._ts_ms)
            if res.face_blendshapes:
                v = np.zeros(len(ARKIT_CHANNELS), dtype=np.float32)
                for cat in res.face_blendshapes[0]:
                    j = idx.get(cat.category_name)
                    if j is not None:
                        v[j] = cat.score
                last = v
            else:
                misses += 1
            rows.append(last.copy())
        if not rows:
            return None, fps, 0.0
        B = np.stack(rows)
        return B, fps, 1.0 - misses / len(rows)


def resample_60fps(B25: np.ndarray, src_fps: float, smooth_ms: float) -> np.ndarray:
    """(T,14)@src_fps → (T',14)@60fps lineer interpolasyon (+ opsiyonel Gaussian)."""
    T = len(B25)
    dur_s = T / src_fps
    t_src = (np.arange(T) + 0.5) / src_fps
    t_dst = np.arange(int(dur_s * OUTPUT_FPS)) / OUTPUT_FPS
    out = np.stack([np.interp(t_dst, t_src, B25[:, c])
                    for c in range(B25.shape[1])], axis=1)
    if smooth_ms > 0:
        sig_f = smooth_ms / (1000.0 / OUTPUT_FPS)
        r = max(1, int(3 * sig_f))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig_f) ** 2)
        k /= k.sum()
        out = np.stack([np.convolve(out[:, c], k, mode="same")
                        for c in range(out.shape[1])], axis=1)
    return out.astype(np.float32)


# ── Ana akış ──────────────────────────────────────────────────────────────────
def process(args):
    from grid_process import detect_time_divisor, parse_align  # aynı klasör şartı

    video_root = Path(args.video).expanduser()
    corpus     = Path(args.corpus).expanduser()
    out_dir    = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ensure_model(Path(args.model).expanduser())

    speakers = args.speakers or [s for s in range(1, 35) if s not in NO_VIDEO_SPEAKERS]
    grand_total, grand_skip = 0, 0

    for spk in speakers:
        spk_dir = None
        for cand in (video_root / f"s{spk}", video_root / f"s{spk}" / "video"):
            if cand.exists():
                spk_dir = video_root / f"s{spk}"
                break
        if spk_dir is None:
            print(f"s{spk:2d}: video klasörü yok — atlanıyor")
            continue

        units = find_units(spk_dir)
        if not units:
            print(f"s{spk:2d}: video birimi bulunamadı — atlanıyor")
            continue

        # align (kelime zamanları) — opsiyonel ama önerilir
        align_dir = corpus / f"s{spk}" / "align"
        divisor = detect_time_divisor(align_dir) if align_dir.exists() else None

        out_file = out_dir / f"s{spk}.jsonl"
        done_ids = set()
        if out_file.exists():
            for line in open(out_file, encoding="utf-8"):
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

        extractor = BlendshapeExtractor(model)
        n_ok, n_skip, det_sum = 0, 0, 0.0
        t0 = time.time()

        with open(out_file, "a", encoding="utf-8") as fout:
            for stem, kind, path in units:
                uid = f"grid_s{spk}_{stem}"
                if uid in done_ids:
                    n_ok += 1
                    continue
                try:
                    B25, fps, det = extractor.extract(kind, path)
                    if B25 is None or det < args.min_detect:
                        n_skip += 1
                        continue
                    det_sum += det
                    A = resample_60fps(B25, fps, args.smooth)

                    words = []
                    ap = align_dir / f"{stem}.align"
                    if divisor and ap.exists():
                        words = [{"w": w, "s": round(s, 4), "e": round(e, 4)}
                                 for w, s, e in parse_align(ap, divisor)]
                    wav = corpus / f"s{spk}" / f"s{spk}" / f"{stem}.wav"

                    rec = {
                        "id":    uid,
                        "lang":  "en",
                        "spk":   f"s{spk}",
                        "src":   "video_gt",
                        "wav":   str(wav.relative_to(corpus)) if wav.exists() else None,
                        "words": words,
                        "fps":      OUTPUT_FPS,
                        "channels": ARKIT_CHANNELS,
                        "frames": [{"t": round(i * 1000.0 / OUTPUT_FPS, 2),
                                    "a": [round(float(x), 3) for x in A[i]]}
                                   for i in range(len(A))],
                        "dur_ms": round(len(B25) / fps * 1000.0, 1),
                        "split":  "val" if spk >= VAL_SPEAKER_MIN else "train",
                        "detect_rate": round(det, 3),
                    }
                    fout.write(json.dumps(rec) + "\n")
                    n_ok += 1
                    if n_ok % 100 == 0:
                        el = time.time() - t0
                        print(f"  s{spk}: {n_ok}/{len(units)}  "
                              f"({el:.0f}s, {el/max(n_ok-len(done_ids),1):.1f}s/utt)")
                except Exception as e:
                    n_skip += 1
                    if n_skip <= 3:
                        print(f"  ⚠ {uid}: {e}")

        rate = det_sum / max(n_ok - len(done_ids), 1)
        print(f"s{spk:2d}: ✅ {n_ok} ok, {n_skip} atlandı "
              f"(ort. tespit {rate:.1%}, {time.time()-t0:.0f}s)")
        grand_total += n_ok; grand_skip += n_skip

    print(f"\n✅ Toplam: {grand_total:,} cümle GT çıkarıldı, {grand_skip} atlandı")

    if args.merge:
        merged = out_dir / "grid_gt_all.jsonl"
        n = 0
        with open(merged, "w", encoding="utf-8") as fout:
            for f in sorted(out_dir.glob("s*.jsonl")):
                for line in open(f, encoding="utf-8"):
                    fout.write(line); n += 1
        print(f"📁 {merged}  ({n:,} kayıt)")
        print("\nEğitim (YENİ cache klasörü kullanın — eski rule-label cache karışmasın):")
        print("  python grid_train.py train --data ...\\grid_gt\\grid_gt_all.jsonl "
              "--corpus ...\\grid_corpus --cache ...\\grid_cache_gt "
              "--runs ...\\grid_runs_gt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GRID video → MediaPipe ARKit GT")
    ap.add_argument("--video",  default="~/grid_video",  help="sN video klasörlerinin kökü")
    ap.add_argument("--corpus", default="~/grid_corpus", help="align + wav kökü")
    ap.add_argument("--out",    default="~/grid_gt")
    ap.add_argument("--model",  default="~/face_landmarker.task")
    ap.add_argument("--speakers", type=int, nargs="*", default=None)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="Opsiyonel Gaussian sigma (ms). 0 = ham GT (önerilen)")
    ap.add_argument("--min-detect", type=float, default=0.80,
                    help="Bu orandan az yüz tespit edilen cümleler atılır")
    ap.add_argument("--merge", action="store_true",
                    help="Bitince sN.jsonl dosyalarını grid_gt_all.jsonl'e birleştir")
    args = ap.parse_args()
    process(args)
