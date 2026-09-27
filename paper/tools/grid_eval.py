#!/usr/bin/env python3
"""
TranscriptionSync — değerlendirme ve makale figürleri (makale 2)
grid_train.py çıktılarından (best.pt + log.csv) offline hesaplanabilen tüm
metrikleri üretir ve IEEE formatında grafik basar.

GEREKSİNİM: grid_train.py ile AYNI klasörde durmalı (model sınıflarını oradan alır).
Bağımlılık: torch + numpy + matplotlib.

Üretilenler (--out klasörüne):
  fig1_training_curves.png/.pdf   eğitim eğrileri (tüm run'lar üst üste)
  fig2_channel_mae.png/.pdf       kanal bazlı MAE çubuk grafiği
  fig3_trajectories_<run>.png     tahmin-vs-hedef yörünge örnekleri
  fig4_lvd_box.png/.pdf           frame başına L2 (LVD-proxy) dağılımı
  fig5_lag_hist.png/.pdf          cümle başına zamansal gecikme histogramı
  summary.csv / summary.md        tüm sayısal sonuçlar (makale tablosu için)

NOT: Bu script SyncNet LSE-C/D, canlı latency ve MOS ÜRETMEZ — onlar render
hattı, canlı sistem enstrümantasyonu ve insan çalışması gerektirir.

Kullanım (Windows):
  python grid_eval.py --cache C:\\Users\\wpu-ai\\grid_cache ^
      --runs C:\\Users\\wpu-ai\\grid_runs C:\\Users\\wpu-ai\\grid_runs_lstm ^
      --out C:\\Users\\wpu-ai\\grid_figures
"""

import csv
import json
import argparse
from pathlib import Path
import numpy as np

FPS = 60
ARKIT_CHANNELS = [
    "jawOpen","mouthClose","mouthFunnel","mouthPucker",
    "mouthStretchLeft","mouthStretchRight",
    "mouthUpperUpLeft","mouthUpperUpRight",
    "mouthLowerDownLeft","mouthLowerDownRight",
    "mouthShrugUpper","mouthRollLower",
    "mouthDimpleLeft","mouthDimpleRight",
]
TRAJ_CHANNELS = ["jawOpen", "mouthClose", "mouthPucker", "mouthStretchLeft"]

# ── IEEE figür stili ──────────────────────────────────────────────────────────
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.0, "axes.grid": True, "grid.alpha": 0.3,
        "figure.constrained_layout.use": True,
    })
    return plt

def _save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    print(f"  🖼  {name}.png/.pdf")

# ── Eğitim logları ────────────────────────────────────────────────────────────
def read_log(run_dir: Path):
    log = run_dir / "log.csv"
    if not log.exists():
        return None
    rows = list(csv.DictReader(open(log)))
    return {
        "epoch":     [int(r["epoch"]) for r in rows],
        "train_mse": [float(r["train_mse"]) for r in rows],
        "val_mse":   [float(r["val_mse"]) for r in rows],
        "val_mae":   [float(r["val_mae"]) for r in rows],
    }

# ── Model değerlendirme ───────────────────────────────────────────────────────
def evaluate_run(run_dir: Path, cache: Path, device: str,
                 max_utts: int, n_samples: int):
    import torch
    from grid_train import BiLSTMModel, CausalTransformer, GridCache

    ck = torch.load(run_dir / "best.pt", map_location="cpu")
    model = CausalTransformer() if ck["arch"] == "transformer" else BiLSTMModel()
    model.load_state_dict(ck["model"])
    dev = torch.device(device)
    model.to(dev).eval()

    ds = GridCache(cache / "val.npz")
    n = min(len(ds), max_utts)

    ch_abs   = np.zeros(len(ARKIT_CHANNELS))   # kanal bazlı |hata| toplamı
    n_frames = 0
    sq_sum   = 0.0
    lvd_all  = []      # frame başına L2 (blendshape uzayı, LVD-proxy)
    lags_ms  = []      # cümle başına çapraz-korelasyon gecikmesi
    vel_pred, vel_tgt = 0.0, 0.0
    samples  = []      # (pred, target) yörünge örnekleri

    jaw = ARKIT_CHANNELS.index("jawOpen")
    max_lag = 30       # ±30 frame = ±500 ms

    with torch.no_grad():
        for i in range(n):
            X, Y = ds[i]
            P = model(X.unsqueeze(0).to(dev))[0].cpu().numpy()
            T_ = Y.numpy()
            err = P - T_
            ch_abs   += np.abs(err).sum(axis=0)
            sq_sum   += (err ** 2).sum()
            n_frames += len(T_)
            lvd_all.append(np.linalg.norm(err, axis=1))
            vel_pred += np.abs(np.diff(P, axis=0)).mean()
            vel_tgt  += np.abs(np.diff(T_, axis=0)).mean()

            # jawOpen çapraz korelasyonuyla zamansal gecikme
            p = P[:, jaw] - P[:, jaw].mean()
            t = T_[:, jaw] - T_[:, jaw].mean()
            if t.std() > 1e-3 and len(t) > 2 * max_lag:
                xc = np.correlate(p, t, mode="full")
                mid = len(xc) // 2
                w = xc[mid - max_lag: mid + max_lag + 1]
                lags_ms.append((np.argmax(w) - max_lag) * 1000.0 / FPS)

            if len(samples) < n_samples:
                samples.append((P, T_))

    lvd = np.concatenate(lvd_all)
    return {
        "name":     run_dir.name,
        "arch":     ck["arch"],
        "val_mse":  sq_sum / (n_frames * len(ARKIT_CHANNELS)),
        "val_mae":  ch_abs.sum() / (n_frames * len(ARKIT_CHANNELS)),
        "ch_mae":   ch_abs / n_frames,
        "lvd_mean": float(lvd.mean()), "lvd_p95": float(np.percentile(lvd, 95)),
        "lvd":      lvd,
        "lag_mean": float(np.mean(lags_ms)) if lags_ms else float("nan"),
        "lag_std":  float(np.std(lags_ms)) if lags_ms else float("nan"),
        "lags":     np.array(lags_ms),
        "vel_ratio": vel_pred / max(vel_tgt, 1e-8),   # 1.0 = hedefle aynı pürüzsüzlük
        "n_utts":   n,
        "samples":  samples,
        "log":      read_log(run_dir),
    }

# ── Figürler ──────────────────────────────────────────────────────────────────
def fig_training_curves(plt, results, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for r in results:
        if r["log"]:
            ax.plot(r["log"]["epoch"], r["log"]["val_mse"],
                    label=f"{r['name']} (val)")
            ax.plot(r["log"]["epoch"], r["log"]["train_mse"],
                    "--", alpha=0.5, label=f"{r['name']} (train)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_yscale("log")
    ax.legend(); ax.set_title("Training curves")
    _save(fig, out, "fig1_training_curves"); plt.close(fig)

def fig_channel_mae(plt, results, out):
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(ARKIT_CHANNELS))
    w = 0.8 / max(len(results), 1)
    for k, r in enumerate(results):
        ax.bar(x + k * w, r["ch_mae"], width=w, label=r["name"])
    ax.set_xticks(x + w * (len(results) - 1) / 2)
    ax.set_xticklabels(ARKIT_CHANNELS, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("MAE"); ax.legend(); ax.set_title("Per-channel MAE (val)")
    _save(fig, out, "fig2_channel_mae"); plt.close(fig)

def fig_trajectories(plt, r, out):
    n = len(r["samples"])
    if n == 0:
        return
    fig, axes = plt.subplots(n, len(TRAJ_CHANNELS),
                             figsize=(7.0, 1.4 * n), sharex="col", squeeze=False)
    for i, (P, T_) in enumerate(r["samples"]):
        t = np.arange(len(T_)) / FPS
        for j, ch in enumerate(TRAJ_CHANNELS):
            c = ARKIT_CHANNELS.index(ch)
            ax = axes[i][j]
            ax.plot(t, T_[:, c], label="target", color="0.3")
            ax.plot(t, P[:, c], "--", label="pred", color="tab:red")
            ax.set_ylim(-0.05, 1.0)
            if i == 0: ax.set_title(ch, fontsize=7)
            if j == 0: ax.set_ylabel(f"utt {i+1}", fontsize=7)
            if i == n - 1: ax.set_xlabel("time (s)")
    axes[0][0].legend(loc="upper right", fontsize=6)
    fig.suptitle(f"Predicted vs target — {r['name']}", fontsize=8)
    _save(fig, out, f"fig3_trajectories_{r['name']}"); plt.close(fig)

def fig_lvd(plt, results, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.boxplot([r["lvd"] for r in results],
               tick_labels=[r["name"] for r in results], showfliers=False)
    ax.set_ylabel("Per-frame L2 (blendshape space)")
    ax.set_title("LVD-proxy distribution (val)")
    _save(fig, out, "fig4_lvd_box"); plt.close(fig)

def fig_lag(plt, results, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for r in results:
        if len(r["lags"]):
            ax.hist(r["lags"], bins=21, range=(-250, 250),
                    alpha=0.55, label=f"{r['name']} (μ={r['lag_mean']:.0f} ms)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Lag (ms)  —  + : tahmin geç")
    ax.set_ylabel("Utterance sayısı"); ax.legend()
    ax.set_title("Temporal lag (jawOpen xcorr)")
    _save(fig, out, "fig5_lag_hist"); plt.close(fig)

# ── Özet tablo ────────────────────────────────────────────────────────────────
def write_summary(results, out: Path):
    cols = ["name","arch","n_utts","val_mse","val_mae",
            "lvd_mean","lvd_p95","lag_mean","lag_std","vel_ratio"]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            w.writerow([r[c] if not isinstance(r[c], float) else f"{r[c]:.5f}"
                        for c in cols])
    lines = ["| Run | Arch | Val MSE | Val MAE | LVD μ | LVD p95 | Lag (ms) | Vel ratio |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['arch']} | {r['val_mse']:.5f} | "
                     f"{r['val_mae']:.4f} | {r['lvd_mean']:.4f} | {r['lvd_p95']:.4f} | "
                     f"{r['lag_mean']:.0f} ± {r['lag_std']:.0f} | {r['vel_ratio']:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n  📄 summary.csv / summary.md → {out}")

# ── Ana akış ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="TranscriptionSync eval + figures")
    ap.add_argument("--cache", default="~/grid_cache")
    ap.add_argument("--runs", nargs="+", default=["~/grid_runs"],
                    help="Bir veya birden çok run klasörü (best.pt + log.csv)")
    ap.add_argument("--out", default="~/grid_figures")
    ap.add_argument("--max-utts", type=int, default=500,
                    help="Val'den değerlendirilecek en çok cümle (hız için)")
    ap.add_argument("--samples", type=int, default=3,
                    help="Yörünge grafiği örnek cümle sayısı")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = Path(args.cache).expanduser()
    out   = Path(args.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

    results = []
    for rd in args.runs:
        rd = Path(rd).expanduser()
        if not (rd / "best.pt").exists():
            print(f"⚠️  {rd}: best.pt yok — atlanıyor"); continue
        print(f"▶ Değerlendiriliyor: {rd.name}  (device={device})")
        results.append(evaluate_run(rd, cache, device,
                                    args.max_utts, args.samples))
    if not results:
        print("❌ Değerlendirilecek run bulunamadı."); return

    plt = _style()
    print("\nFigürler:")
    fig_training_curves(plt, results, out)
    fig_channel_mae(plt, results, out)
    for r in results:
        fig_trajectories(plt, r, out)
    fig_lvd(plt, results, out)
    fig_lag(plt, results, out)
    print()
    write_summary(results, out)

if __name__ == "__main__":
    main()
