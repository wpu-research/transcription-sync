# Atlas — Makale 2 (TranscriptionSync / GRID) İş Akışı Notları

Bu notlar, `paper/tools/` altındaki tüm GRID/eğitim/değerlendirme/makale
script'lerine dair konuşma özetidir. Tüm script'ler `paper/tools/` içinde
ve birbirine bağımlı (aynı klasörde durmalı: align parser, model sınıfları,
ARKIT_CHANNELS sabiti birbirinden import edilir).

## Genel Boru Hattı (sırasıyla)

1. `grid_download.py` — videoları indir (+ opsiyonel extract)
2. `grid_process.py` — GRID corpus (.align + .wav) → kural-tabanlı
   viseme→ARKit eğitim seti (`grid_dataset/grid_all.jsonl`)
3. `grid_video_gt.py` — video → MediaPipe gerçek ARKit GT
   (`grid_gt/grid_gt_all.jsonl`) — **2'nin alternatifi/iyileştirmesi**,
   aynı JSONL formatı
4. `grid_train.py` — audio (log-mel) → 14 ARKit kanal model eğitimi
   (LSTM veya Transformer)
5. `grid_eval.py` — eğitim çıktılarından makale figürleri + metrikler
6. `build_paper_docx.py` — `ATLAS_IEEE_Access_Draft.md` → `.docx`

---

## 0) `grid_process.py` — Kural-tabanlı eğitim seti (Makale 2)

```powershell
python grid_process.py --corpus ~/grid_corpus --output ~/grid_dataset
```

- Yalnızca numpy gerekir, video gerekmez (sadece `.align` + `.wav`).
- GRID `.align` zaman birimi otomatik tespit edilir (25 kHz örnek →
  saniye = değer/25000), 1000x birim hatası önlenir.
- Harf token'ları fonem adına çevrilir (b → "bee" → B IY).
- 14 viseme sınıfı → `VISEME_TO_ARKIT` ile 14 gerçek ARKit mouth
  kanalına dönüştürülür (viseme indeksi de saklanır).
- Split: s1–s28 train, s29–s34 val (speaker-disjoint, makale V-D ile
  uyumlu).
- **Not:** Bu, makale-1'in V-D validation bölümünde KULLANILMAZ — orada
  ham `.align` + MFA kullanılır. Bu sadece makale-2 (öğrenilmiş model)
  eğitim seti.
- `grid_video_gt.py` ile üretilen GT, bu kural-tabanlı setin
  **gerçek (MediaPipe ölçümü) alternatifidir** — aynı JSONL formatı,
  `grid_train.py` değişmeden çalışır.

---

## 1) GRID Video → MediaPipe GT Çıkarımı (`grid_video_gt.py`)

### a) Video indirme (Windows, `video_download.py` / `grid_download.py`)

```powershell
python video_download.py --out C:\Users\wpu-ai\Desktop\Atlas_Journal\grid_video
```

- Kaynak: Zenodo. 33 konuşmacı (s21 hariç — videosu yok).
- Resume destekli; zip'ler indirilir ama bu komut **--extract olmadan
  açılmaz**.
- `s7` gibi tek tek konuşmacılarda 504 Gateway Timeout görülebilir,
  script otomatik tekrar dener.

### b) Zip'leri açma

```powershell
python video_download.py --out C:\Users\wpu-ai\Desktop\Atlas_Journal\grid_video --extract
```

Tamamlanmış indirmeler atlanır, sadece açma işlemi yapılır.

### c) MediaPipe GT çıkarımı (`grid_video_gt.py`)

## 1) Video indirme (Windows, `video_download.py` / `grid_download.py`)

```powershell
python video_download.py --out C:\Users\wpu-ai\Desktop\Atlas_Journal\grid_video
```

- Kaynak: Zenodo. 33 konuşmacı (s21 hariç — videosu yok).
- Resume destekli; zip'ler indirilir ama bu komut **--extract olmadan
  açılmaz**.
- `s7` gibi tek tek konuşmacılarda 504 Gateway Timeout görülebilir,
  script otomatik tekrar dener.

## 2) Zip'leri açma

`--extract` flag'i ile (veya aynı komutu tekrar çalıştırıp açma adımını
eklemek için):

```powershell
python video_download.py --out C:\Users\wpu-ai\Desktop\Atlas_Journal\grid_video --extract
```

Tamamlanmış indirmeler atlanır, sadece açma işlemi yapılır.

## 3) MediaPipe GT çıkarımı (`grid_video_gt.py`)

Repo konumu: `paper/tools/grid_video_gt.py`
Bağımlı dosya (aynı klasörde olmalı): `paper/tools/grid_process.py`
(align parser ve `detect_time_divisor` buradan import edilir)

Gerekli paketler:

```powershell
pip install mediapipe opencv-python
```

Çalıştırma:

```powershell
python grid_video_gt.py --video C:\Users\wpu-ai\Desktop\Atlas_Journal\grid_video --corpus C:\Users\wpu-ai\Desktop\Atlas_Journal\grid_corpus --out C:\Users\wpu-ai\Desktop\Atlas_Journal\grid_gt --merge
```

Notlar:
- `face_landmarker.task` modeli (`~4 MB`) yoksa otomatik indirilir
  (`C:\Users\wpu-ai\face_landmarker.task`).
- Video 25 fps → blendshape'ler 60 fps'e lineer interpolasyonla
  örneklenir.
- Konuşmacı başına `sN.jsonl` yazılır; kesilirse aynı komutla devam
  eder (işlenen id'ler atlanır).
- Yüz tespiti %80 altında kalan cümleler atılır ve raporlanır
  (`--min-detect` ile ayarlanabilir).
- `--merge`: sonunda tüm `sN.jsonl` dosyalarını `grid_gt_all.jsonl`'de
  birleştirir → bu dosya `grid_train.py`'a verilir.
- s21 videosu yok, otomatik atlanır.

## 2) CPU vs GPU sorusu (grid_video_gt.py)

- MediaPipe Tasks API (`face_landmarker.task`) Python tarafında
  varsayılan olarak **CPU (XNNPACK delegate)** kullanır.
- GPU delegate, Tasks API'de Windows Python için resmi/olgun değil —
  bu yüzden script delegate belirtmiyor.
- Model çok küçük (~4 MB, tek yüz), CPU'da bile cümle başına ~1-2 sn.
  Asıl darboğaz video decode/IO, GPU inference değil.
- Hız için gerçek kazanç: **multiprocessing ile konuşmacıları paralel
  işlemek** olurdu (şu an script seri/tek-process). İstenirse bu
  eklenebilir — önce mevcut hız (ilerleme satırları) gözlemlenip
  gerçekten gerekli mi karar verilecek.

## 3) `grid_train.py` — Model eğitimi (Makale 2)

Audio (log-mel) → 14 kanal ARKit blendshape @ 60 fps.
Bağımlılık: torch + numpy (librosa/torchaudio GEREKMEZ — wav stdlib
`wave`, log-mel numpy ile hesaplanır).

```powershell
# 1) Tek seferlik cache (JSONL + wav → NPZ, ~50x hızlı yükleme)
python grid_train.py prepare --data C:\...\grid_gt\grid_gt_all.jsonl --corpus C:\...\grid_corpus --cache C:\...\grid_cache_gt

# 2) Eğitim (cache yoksa otomatik prepare çalışır)
python grid_train.py train --data C:\...\grid_gt\grid_gt_all.jsonl --corpus C:\...\grid_corpus --cache C:\...\grid_cache_gt --arch transformer --epochs 30 --runs C:\...\grid_runs_gt

# (opsiyonel) BiLSTM baseline (Mimari B) — varsayılan transformer (Mimari A)
python grid_train.py train ... --arch lstm --runs C:\...\grid_runs_gt_lstm

# 3) Tek wav üzerinde çıkarım
python grid_train.py infer --checkpoint C:\...\grid_runs_gt\best.pt --wav input.wav --out frames.json
```

**ÖNEMLİ:** GT video çıkarımı (`grid_video_gt.py`) ile eğitirken, eski
kural-tabanlı (`grid_process.py`) cache ile karışmaması için **yeni bir
cache klasörü** kullanılmalı (`grid_cache_gt` gibi).

---

## 4) `grid_eval.py` — Değerlendirme ve makale figürleri

`grid_train.py` ile AYNI klasörde durmalı (model sınıflarını oradan
import eder). Bağımlılık: torch + numpy + matplotlib.

```powershell
python grid_eval.py --cache C:\...\grid_cache_gt --runs C:\...\grid_runs_gt C:\...\grid_runs_gt_lstm --out C:\...\grid_figures
```

Üretilenler (`--out` klasörüne):
- `fig1_training_curves` — eğitim eğrileri (tüm run'lar üst üste)
- `fig2_channel_mae` — kanal bazlı MAE çubuk grafiği
- `fig3_trajectories_<run>` — tahmin-vs-hedef yörünge örnekleri
- `fig4_lvd_box` — frame başına L2 (LVD-proxy) dağılımı
- `fig5_lag_hist` — cümle başına zamansal gecikme histogramı
- `summary.csv` / `summary.md` — makale tablosu için tüm sayısal sonuçlar

**Üretmediği şeyler:** SyncNet LSE-C/D, canlı latency, MOS — bunlar
render hattı, canlı sistem enstrümantasyonu ve insan çalışması
gerektirir.

Mevcut `train_results/grid_figures/` ve `train_results/grid_runs*/`
klasörleri kural-tabanlı (`grid_process.py`) veriyle üretilmiş ilk
sonuçlardır — GT (video) sonuçları ayrı klasörlere (`grid_runs_gt` vb.)
yazılmalı, eskiler üzerine yazılmamalı.

---

## 5) `build_paper_docx.py` — Makaleyi Word'e çevir

```powershell
python build_paper_docx.py
```

- Girdi: `paper/ATLAS_IEEE_Access_Draft.md`
- Çıktı: `paper/ATLAS_IEEE_Access_Draft.docx`
- IEEE Access stiline göre biçimlendirme (navy/blue başlıklar, tablo
  shading vb.) uygular.
- Yeni figürler/sonuçlar eklendiğinde önce `.md` güncellenmeli, sonra
  bu script tekrar çalıştırılmalı.

---

## Genel Sıradaki Adımlar

1. GT çıkarımı bitince → `grid_train.py prepare` + `train` (yeni cache
   klasörüyle, transformer + lstm ikisi de)
2. → `grid_eval.py` ile figürleri ve `summary.md`'yi üret
3. → Sonuçları `ATLAS_IEEE_Access_Draft.md`'e işle
4. → `build_paper_docx.py` ile `.docx` güncelle
