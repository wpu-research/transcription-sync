#!/usr/bin/env python3
"""
TranscriptionSync — GRID eğitim scripti (makale 2: öğrenilmiş model)
Audio (log-mel) → 14 kanal ARKit blendshape @ 60 fps

Bağımlılık: torch + numpy (librosa/torchaudio GEREKMEZ; wav okuma stdlib `wave`,
log-mel numpy ile hesaplanır). Windows/Linux/Mac.

Mimari:
  --arch lstm         BiLSTM baseline   (görseldeki Mimari B)
  --arch transformer  Causal Transformer (görseldeki Mimari A, varsayılan)

Kullanım (Windows):
  # 1) Tek seferlik cache (JSONL + wav → NPZ, ~50x hızlı yükleme):
  python grid_train.py prepare --data C:\\Users\\wpu-ai\\grid_dataset\\grid_all.jsonl --corpus C:\\Users\\wpu-ai\\grid_corpus --cache C:\\Users\\wpu-ai\\grid_cache

  # 2) Eğitim (cache yoksa otomatik prepare çalışır):
  python grid_train.py train --data ... --corpus ... --cache ... --arch transformer --epochs 30

  # 3) Tek wav üzerinde çıkarım:
  python grid_train.py infer --checkpoint C:\\...\\runs\\best.pt --wav input.wav --out frames.json
"""

import os, json, math, time, wave, argparse, csv
from pathlib import Path
import numpy as np

# ── Sabitler (grid_process.py ile uyumlu) ─────────────────────────────────────
OUTPUT_FPS = 60
N_MELS     = 80
N_ARKIT    = 14
ARKIT_CHANNELS = [
    "jawOpen","mouthClose","mouthFunnel","mouthPucker",
    "mouthStretchLeft","mouthStretchRight",
    "mouthUpperUpLeft","mouthUpperUpRight",
    "mouthLowerDownLeft","mouthLowerDownRight",
    "mouthShrugUpper","mouthRollLower",
    "mouthDimpleLeft","mouthDimpleRight",
]

# ════════════════════════════════════════════════════════════════════════════
# 1) SES → LOG-MEL (numpy, bağımlılıksız)
# ════════════════════════════════════════════════════════════════════════════
def read_wav(path: Path):
    """16-bit PCM wav → float32 [-1,1] mono + örnekleme hızı."""
    with wave.open(str(path), "rb") as wf:
        sr   = wf.getframerate()
        nch  = wf.getnchannels()
        sw   = wf.getsampwidth()
        data = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: yalnızca 16-bit PCM desteklenir (sampwidth={sw})")
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x, sr

def _mel_filterbank(sr, n_fft, n_mels, fmin=40.0):
    fmax = sr / 2.0
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    freqs = mel2hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        if c == lo: c += 1
        if hi == c: hi += 1
        fb[m - 1, lo:c]  = (np.arange(lo, c) - lo) / max(c - lo, 1)
        fb[m - 1, c:hi]  = (hi - np.arange(c, hi)) / max(hi - c, 1)
    return fb

_FB_CACHE = {}
def log_mel_60fps(x: np.ndarray, sr: int) -> np.ndarray:
    """60 fps'e hizalı log-mel: hop = sr/60. Çıkış (T, N_MELS) float32."""
    hop   = int(round(sr / OUTPUT_FPS))
    n_fft = 1024 if sr <= 30000 else 2048
    key = (sr, n_fft)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(sr, n_fft, N_MELS)
    fb = _FB_CACHE[key]
    win = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    x = np.pad(x, (pad, pad))
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames < 1:
        return np.zeros((0, N_MELS), dtype=np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel  = spec @ fb.T
    logm = np.log(mel + 1e-5).astype(np.float32)
    # utterance bazında normalizasyon
    logm = (logm - logm.mean()) / (logm.std() + 1e-5)
    return logm

# ════════════════════════════════════════════════════════════════════════════
# 2) PREPARE: JSONL + wav → NPZ cache
# ════════════════════════════════════════════════════════════════════════════
def prepare(data_jsonl: Path, corpus: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    buf = {"train": {"X": [], "Y": [], "L": []},
           "val":   {"X": [], "Y": [], "L": []}}
    n_ok, n_skip = 0, 0
    t0 = time.time()

    with open(data_jsonl, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            try:
                r = json.loads(line)
                if not r.get("wav"):
                    n_skip += 1; continue
                wav_path = corpus / r["wav"]
                if not wav_path.exists():
                    n_skip += 1; continue

                x, sr = read_wav(wav_path)
                mel = log_mel_60fps(x, sr)                       # (Tm, 80)
                tgt = np.array([fr["a"] for fr in r["frames"]],
                               dtype=np.float32)                  # (Tf, 14)
                T = min(len(mel), len(tgt))
                if T < 30:                                        # < 0.5 s → atla
                    n_skip += 1; continue

                split = r.get("split", "train")
                buf[split]["X"].append(mel[:T].astype(np.float16))
                buf[split]["Y"].append(tgt[:T].astype(np.float16))
                buf[split]["L"].append(T)
                n_ok += 1
                if n_ok % 2000 == 0:
                    print(f"  {n_ok:6d} sample işlendi ({time.time()-t0:.0f}s)")
            except Exception:
                n_skip += 1

    for split, d in buf.items():
        if not d["L"]:
            print(f"  ⚠️  {split}: boş"); continue
        X = np.concatenate(d["X"], axis=0)
        Y = np.concatenate(d["Y"], axis=0)
        L = np.array(d["L"], dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(L)])
        np.savez(cache_dir / f"{split}.npz", X=X, Y=Y, L=L, off=off)
        print(f"  💾 {split}: {len(L):,} utt | {X.shape[0]:,} frame | "
              f"{(X.nbytes + Y.nbytes)/1e6:.0f} MB")
    print(f"✅ prepare bitti: {n_ok:,} ok, {n_skip:,} atlandı ({time.time()-t0:.0f}s)")

# ════════════════════════════════════════════════════════════════════════════
# 3) MODELLER
# ════════════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    """Mimari B — BiLSTM baseline (non-causal)."""
    def __init__(self, d_in=N_MELS, hidden=192, layers=2, d_out=N_ARKIT):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, d_out))
    def forward(self, x, mask=None):
        h, _ = self.lstm(x)
        return torch.sigmoid(self.head(h))

class CausalTransformer(nn.Module):
    """Mimari A — causal Transformer (gerçek zamanlı dağıtıma uygun)."""
    def __init__(self, d_in=N_MELS, d_model=192, heads=4, layers=4,
                 d_ff=384, d_out=N_ARKIT, max_len=4096):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(d_model, heads, d_ff, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, d_out)
    def forward(self, x, mask=None):
        T = x.size(1)
        h = self.proj(x) + self.pe[:, :T]
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool),
                            diagonal=1)
        h = self.enc(h, mask=causal, src_key_padding_mask=mask)
        return torch.sigmoid(self.head(h))

# ════════════════════════════════════════════════════════════════════════════
# 4) DATASET / LOSS / TRAIN
# ════════════════════════════════════════════════════════════════════════════
class GridCache(torch.utils.data.Dataset):
    def __init__(self, npz_path: Path):
        d = np.load(npz_path)
        self.X, self.Y = d["X"], d["Y"]
        self.off = d["off"]
        self.n = len(d["L"])
    def __len__(self): return self.n
    def __getitem__(self, i):
        s, e = self.off[i], self.off[i+1]
        return (torch.from_numpy(self.X[s:e].astype(np.float32)),
                torch.from_numpy(self.Y[s:e].astype(np.float32)))

def collate(batch):
    Ls = [x.size(0) for x, _ in batch]
    T  = max(Ls); B = len(batch)
    X = torch.zeros(B, T, N_MELS); Y = torch.zeros(B, T, N_ARKIT)
    pad = torch.ones(B, T, dtype=torch.bool)         # True = padding
    for i, (x, y) in enumerate(batch):
        X[i, :x.size(0)] = x; Y[i, :y.size(0)] = y; pad[i, :x.size(0)] = False
    return X, Y, pad

def masked_loss(pred, tgt, pad, w_vel=0.5):
    """MSE + hız (Δframe) kaybı — dudak hareketi pürüzsüzlüğü için."""
    m = (~pad).float().unsqueeze(-1)
    mse = ((pred - tgt) ** 2 * m).sum() / (m.sum() * N_ARKIT + 1e-8)
    dp, dt = pred[:, 1:] - pred[:, :-1], tgt[:, 1:] - tgt[:, :-1]
    mv = m[:, 1:] * m[:, :-1]
    vel = ((dp - dt) ** 2 * mv).sum() / (mv.sum() * N_ARKIT + 1e-8)
    return mse + w_vel * vel, mse.item()

def train(args):
    cache = Path(args.cache).expanduser()
    if not (cache / "train.npz").exists():
        print("Cache yok — önce prepare çalışıyor…")
        prepare(Path(args.data).expanduser(), Path(args.corpus).expanduser(), cache)

    dev = torch.device(args.device if args.device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {dev}")
    tr = GridCache(cache / "train.npz")
    va = GridCache(cache / "val.npz")
    print(f"Data   : train {len(tr):,} utt | val {len(va):,} utt")
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                        collate_fn=collate, num_workers=args.workers,
                                        pin_memory=(dev.type == "cuda"))
    dl_va = torch.utils.data.DataLoader(va, batch_size=args.batch, shuffle=False,
                                        collate_fn=collate, num_workers=args.workers)

    model = (CausalTransformer() if args.arch == "transformer"
             else BiLSTMModel()).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Model  : {args.arch} ({n_par/1e6:.2f}M parametre)")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    run   = Path(args.runs).expanduser(); run.mkdir(parents=True, exist_ok=True)
    logf  = open(run / "log.csv", "w", newline="")
    logw  = csv.writer(logf); logw.writerow(["epoch","train_mse","val_mse","val_mae","lr","sec"])

    best, patience = float("inf"), 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train(); tr_mse, nb = 0.0, 0
        for X, Y, pad in dl_tr:
            X, Y, pad = X.to(dev), Y.to(dev), pad.to(dev)
            loss, mse = masked_loss(model(X, pad), Y, pad)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tr_mse += mse; nb += 1
        sched.step()

        model.eval(); va_mse, va_mae, vm = 0.0, 0.0, 0
        with torch.no_grad():
            for X, Y, pad in dl_va:
                X, Y, pad = X.to(dev), Y.to(dev), pad.to(dev)
                P = model(X, pad); m = (~pad).float().unsqueeze(-1)
                va_mse += (((P - Y) ** 2 * m).sum() / (m.sum() * N_ARKIT)).item()
                va_mae += (((P - Y).abs() * m).sum() / (m.sum() * N_ARKIT)).item()
                vm += 1
        va_mse /= max(vm, 1); va_mae /= max(vm, 1); tr_mse /= max(nb, 1)
        sec = time.time() - t0
        lr_now = sched.get_last_lr()[0]
        print(f"ep {ep:3d} | train {tr_mse:.5f} | val {va_mse:.5f} "
              f"(MAE {va_mae:.4f}) | lr {lr_now:.2e} | {sec:.0f}s")
        logw.writerow([ep, f"{tr_mse:.6f}", f"{va_mse:.6f}",
                       f"{va_mae:.6f}", f"{lr_now:.2e}", f"{sec:.0f}"]); logf.flush()

        if va_mse < best - 1e-6:
            best, patience = va_mse, 0
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "channels": ARKIT_CHANNELS, "n_mels": N_MELS,
                        "fps": OUTPUT_FPS, "val_mse": best},
                       run / "best.pt")
            print(f"        💾 best.pt güncellendi (val {best:.5f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stop (patience={args.patience})."); break

    logf.close()
    print(f"\n✅ Eğitim bitti. En iyi val MSE: {best:.5f}")
    print(f"📁 Checkpoint: {run/'best.pt'}  |  Log: {run/'log.csv'}")

# ════════════════════════════════════════════════════════════════════════════
# 5) INFER: tek wav → 60 fps ARKit frame JSON
# ════════════════════════════════════════════════════════════════════════════
def infer(args):
    dev = torch.device("cpu")
    ck = torch.load(Path(args.checkpoint).expanduser(), map_location=dev)
    model = (CausalTransformer() if ck["arch"] == "transformer"
             else BiLSTMModel())
    model.load_state_dict(ck["model"]); model.eval()
    x, sr = read_wav(Path(args.wav).expanduser())
    mel = torch.from_numpy(log_mel_60fps(x, sr)).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        P = model(mel)[0].numpy()
    dt = time.time() - t0
    frames = [{"t": round(i * 1000.0 / OUTPUT_FPS, 2),
               "a": [round(float(v), 3) for v in P[i]]}
              for i in range(len(P))]
    out = Path(args.out).expanduser()
    out.write_text(json.dumps({"fps": OUTPUT_FPS, "channels": ck["channels"],
                               "frames": frames}, indent=None), encoding="utf-8")
    audio_s = len(x) / sr
    print(f"✅ {len(frames)} frame → {out}")
    print(f"⏱️  {dt*1000:.0f} ms inference / {audio_s:.1f} s ses "
          f"(RTF {dt/audio_s:.3f})")

# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TranscriptionSync GRID eğitimi")
    sub = ap.add_subparsers(dest="cmd")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data",   default="~/grid_dataset/grid_all.jsonl")
    common.add_argument("--corpus", default="~/grid_corpus")
    common.add_argument("--cache",  default="~/grid_cache")

    sub.add_parser("prepare", parents=[common])

    tp = sub.add_parser("train", parents=[common])
    tp.add_argument("--arch", choices=["transformer", "lstm"], default="transformer")
    tp.add_argument("--epochs", type=int, default=30)
    tp.add_argument("--batch", type=int, default=32)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--patience", type=int, default=6)
    tp.add_argument("--workers", type=int, default=0)   # Windows'ta 0 güvenli
    tp.add_argument("--device", default=None, help="cuda | cpu (otomatik)")
    tp.add_argument("--runs", default="~/grid_runs")

    ip = sub.add_parser("infer")
    ip.add_argument("--checkpoint", required=True)
    ip.add_argument("--wav", required=True)
    ip.add_argument("--out", default="frames.json")

    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(Path(args.data).expanduser(), Path(args.corpus).expanduser(),
                Path(args.cache).expanduser())
    elif args.cmd == "infer":
        infer(args)
    else:
        if args.cmd is None:
            args = ap.parse_args(["train"])
        train(args)
