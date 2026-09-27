"""Generate Figures 1-4 for the IEEE Access draft from the paper's reported numbers.

Figure 1: system architecture block diagram (Section III).
Figure 2: Table 4 synchronization results (Section V-C).
Figure 3: Table 5 layer-wise GRID validation (Section V-D).
Figure 4: Table 6 latency / CPU cost, EN vs TR (Section V-E).

Output: paper/figures/fig{1..4}.pdf and .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: System architecture
# ---------------------------------------------------------------------------
def box(ax, xy, w, h, text, fc="white", ec="black", fontsize=8.5):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                        linewidth=1.1, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, wrap=True)
    return (x, y, w, h)


def arrow(ax, b1, b2, side1="right", side2="left", text=None, style="-|>", color="black"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {
        "right": (x1 + w1, y1 + h1 / 2),
        "left": (x1, y1 + h1 / 2),
        "top": (x1 + w1 / 2, y1 + h1),
        "bottom": (x1 + w1 / 2, y1),
    }
    pts2 = {
        "right": (x2 + w2, y2 + h2 / 2),
        "left": (x2, y2 + h2 / 2),
        "top": (x2 + w2 / 2, y2 + h2),
        "bottom": (x2 + w2 / 2, y2),
    }
    a = FancyArrowPatch(pts[side1], pts2[side2], arrowstyle=style,
                         mutation_scale=10, linewidth=1.0, color=color)
    ax.add_patch(a)
    if text:
        mx, my = (pts[side1][0] + pts2[side2][0]) / 2, (pts[side1][1] + pts2[side2][1]) / 2
        ax.text(mx, my + 0.06, text, ha="center", va="bottom", fontsize=7, color=color)


def route(ax, pts, text=None, color="black", style="-|>", label_xy=None):
    """Draw an orthogonal (elbow) connector through `pts`, arrowhead on the last
    segment. Keeps wires axis-aligned so the diagram reads without crossings."""
    for k in range(len(pts) - 2):
        ax.plot([pts[k][0], pts[k + 1][0]], [pts[k][1], pts[k + 1][1]],
                color=color, linewidth=1.0, solid_capstyle="round", zorder=1)
    a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle=style, mutation_scale=10,
                         linewidth=1.0, color=color, zorder=1)
    ax.add_patch(a)
    if text and label_xy:
        ax.text(label_xy[0], label_xy[1], text, fontsize=7, color=color,
                ha="center", va="center")


fig, ax = plt.subplots(figsize=(12.2, 7.6))
ax.set_xlim(0, 15.6)
ax.set_ylim(0, 10.7)
ax.axis("off")

# --- Left spine: audio I/O (all share x = 3.0, so wires are vertical) ---------
mic = box(ax, (3.0, 9.1), 2.6, 1.0, "Microphone\n16 kHz\n(listen_audio)")
gemini = box(ax, (3.0, 6.9), 2.6, 1.5, "Gemini 2.5 Flash\nNative Audio\n(Live API session)",
             fc="#eaf2ff")
playback = box(ax, (3.0, 4.8), 2.6, 1.2, "Audio playback\n24 kHz\n(play_audio,\nsounddevice)")
envelope = box(ax, (3.0, 3.0), 2.6, 1.0, "Amplitude\nenvelope (RMS)")
memory = box(ax, (0.2, 7.0), 2.5, 1.3, "memory/\nlong_term.json", fc="#f5f5f5")

# --- Top lane: tool orchestration (all share the gemini row, y ~ 7.0) --------
toolcall = box(ax, (6.4, 7.0), 2.5, 1.3, "Tool call\n(send_realtime /\nreceive_audio)")
router = box(ax, (9.3, 7.0), 2.5, 1.3, "Tool router\n(negative routing\nconstraints)")
modules = box(ax, (12.2, 6.9), 3.0, 1.5, "19 action modules\n(memory, planner-\nexecutor agent,\ngame mgmt, ...)",
              fc="#eef7ea")
agent = box(ax, (12.2, 9.0), 3.0, 1.1, "agent/\nplanner + executor", fc="#f5f5f5")

# --- Bottom lane: viseme pipeline + presentation (share y ~ 3.0) -------------
transcript = box(ax, (6.4, 3.0), 2.5, 1.3, "Output\ntranscription\n(word timestamps)")
viseme = box(ax, (9.3, 3.0), 2.5, 1.3, "Viseme worker\n(TranscriptionSync\nLayers 1-3)", fc="#fff3e0")
blend = box(ax, (12.2, 3.0), 3.0, 1.3, "14-ch ARKit\nblendshape\nstream @ 60 fps")
ws = box(ax, (12.2, 0.7), 3.0, 1.0, "WebSocket\n(port 7862)")
browser = box(ax, (8.6, 0.5), 3.2, 1.6,
              "Browser UI (HTTP :7861)\nThree.js + @pixiv/three-vrm\n"
              "52 ARKit blendshapes\nemotion, auto-blink",
              fc="#f3e8ff")

# --- Straight, axis-aligned wires --------------------------------------------
arrow(ax, mic, gemini, "bottom", "top")
arrow(ax, gemini, playback, "bottom", "top", text="audio out")
arrow(ax, playback, envelope, "bottom", "top")
arrow(ax, memory, gemini, "right", "left", style="<|-|>", color="#777777")

arrow(ax, gemini, toolcall, "right", "left", text="function call")
arrow(ax, toolcall, router, "right", "left")
arrow(ax, router, modules, "right", "left")
arrow(ax, agent, modules, "bottom", "top", style="<|-|>", color="#777777")

arrow(ax, transcript, viseme, "right", "left")
arrow(ax, viseme, blend, "right", "left")
arrow(ax, blend, ws, "bottom", "top")
arrow(ax, ws, browser, "left", "right")

# --- Two elbow routes that would otherwise cross boxes ------------------------
# gemini -> transcript: down the clear corridor between the audio column and
# the tool lane (x = 6.0), then into transcript's left edge.
route(ax, [(5.6, 7.4), (6.0, 7.4), (6.0, 3.65), (6.4, 3.65)],
      text="transcript", label_xy=(6.0, 5.5))
# amplitude envelope -> viseme worker: a lane just below the box row (y = 2.6),
# clearing the browser panel beneath it.
route(ax, [(4.3, 3.0), (4.3, 2.6), (10.55, 2.6), (10.55, 3.0)],
      text="amplitude\nenvelope", label_xy=(7.6, 2.35))

ax.set_title("A.T.L.A.S system architecture", fontsize=12, pad=12)
save(fig, "fig1_architecture")

# ---------------------------------------------------------------------------
# Figure 2: Table 4 — synchronization results (LVD and SyncNet LSE-C)
# ---------------------------------------------------------------------------
conditions = ["AMP", "RATE", "TSYNC-dur", "TSYNC-coart", "TSYNC\n(ours)", "MFA\n(oracle)"]
lvd = [0.520, 0.556, 0.461, 0.645, 0.461, np.nan]
lvd_sd = [0.029, 0.030, 0.062, 0.046, 0.065, 0.0]
lsec = [2.88, 0.79, 1.39, 0.85, 1.36, 2.90]
lsec_sd = [0.35, 0.32, 0.32, 0.24, 0.29, 0.56]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
x = np.arange(len(conditions))
colors = ["#9e9e9e", "#9e9e9e", "#4a90d9", "#4a90d9", "#1f5fa8", "#2e7d32"]

ax = axes[0]
mask = ~np.isnan(lvd)
ax.bar(x[mask], np.array(lvd)[mask], yerr=np.array(lvd_sd)[mask],
       color=np.array(colors)[mask], capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylabel("LVD (lower is better)")
ax.set_title("(a) Lip vertex distance")

ax = axes[1]
ax.bar(x, lsec, yerr=lsec_sd, color=colors, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=7.5)
ax.set_ylabel("SyncNet LSE-C (higher is better)")
ax.set_title("(b) SyncNet confidence")

fig.suptitle("Objective synchronization by condition (N=50 rendered utterances, Table 4)")
fig.tight_layout()
save(fig, "fig2_synchronization")

# ---------------------------------------------------------------------------
# Figure 3: Table 5 — layer-wise GRID validation
# ---------------------------------------------------------------------------
labels = ["Layer 2\n(ref. words,\nλ priors)", "Layer 2\n(ref. words,\nuniform split)",
          "Layer 1\n(word start,\ntiny)", "Layer 1+2\n(ASR words,\nλ priors)"]
values = [18, 20, 69, 65]
errs = [16, 20, 68, 54]
bound = 45

fig, ax = plt.subplots(figsize=(6, 3.6))
x = np.arange(len(labels))
bars = ax.bar(x, values, yerr=errs, capsize=3, color=["#2e7d32", "#9e9e9e", "#9e9e9e", "#1f5fa8"])
ax.axhline(bound, color="red", linestyle="--", linewidth=1)
ax.text(len(labels) - 0.5, bound + 2, "±45 ms detectability bound", color="red",
        fontsize=8, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Timing error, mean (SD), ms")
ax.set_title("Layer-wise timing validation on GRID (Table 5)")
fig.tight_layout()
save(fig, "fig3_layerwise_grid")

# ---------------------------------------------------------------------------
# Figure 4: Table 6 — latency / CPU, EN vs TR
# ---------------------------------------------------------------------------
metrics = ["Word emission\nlag, median (ms)", "Word emission\nlag, P95 (ms)",
           "ASR window\nproc., median (ms)", "ASR window\nproc., P95 (ms)"]
en = [0.5, 214, 479, 512]  # EN word-emission median is 0.0 ms; floored to 0.5 for log display
tr = [902, 4915, 1779, 5335]

fig, ax = plt.subplots(figsize=(6.5, 3.6))
x = np.arange(len(metrics))
w = 0.35
bars_en = ax.bar(x - w / 2, en, w, label="English", color="#1f5fa8")
ax.bar(x + w / 2, tr, w, label="Turkish", color="#d9822b")
ax.text(bars_en[0].get_x() + bars_en[0].get_width() / 2, 0.6, "0",
        ha="center", va="bottom", fontsize=8)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=8)
ax.set_ylabel("Time (ms, log scale)")
ax.set_title("Layer 1 latency: English vs. Turkish (Table 6)")
ax.legend()
fig.tight_layout()
save(fig, "fig4_latency_en_tr")

# ---------------------------------------------------------------------------
# Figure 5: auxiliary regression architectures (Section V-F)
#   Architecture A — causal Transformer encoder
#   Architecture B — BiLSTM
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.4))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 10)
ax.axis("off")

# Shared input (centered above both columns)
inp = box(ax, (4.95, 8.4), 2.6, 0.8, "Log-mel audio\n(T x 80, 60 fps)", fc="#f5f5f5")

# --- Architecture A: causal Transformer (left column) ---
ax.text(2.95, 7.85, "Architecture A — Causal Transformer encoder", fontsize=9,
        ha="center", fontweight="bold")
projA = box(ax, (0.6, 6.6), 4.7, 0.65, "Linear projection: 80 -> d_model=192")
peA = box(ax, (0.6, 5.8), 4.7, 0.65, "+ sinusoidal positional encoding")
encA = box(ax, (0.6, 3.9), 4.7, 1.6,
           "Causal TransformerEncoder\n4 x [self-attn (4 heads, causal mask)\n+ FFN 192->384->192], norm-first",
           fc="#eaf2ff")
headA = box(ax, (0.6, 2.5), 4.7, 0.65, "Linear head: 192 -> 14, sigmoid")
outA = box(ax, (0.6, 1.1), 4.7, 0.8, "b_hat(t) in [0,1]^14\n(ARKit blendshapes, 60 fps)",
           fc="#fff3e0")

arrow(ax, inp, projA, "left", "top")
arrow(ax, projA, peA, "bottom", "top")
arrow(ax, peA, encA, "bottom", "top")
arrow(ax, encA, headA, "bottom", "top")
arrow(ax, headA, outA, "bottom", "top")

# --- Architecture B: BiLSTM (right column) ---
ax.text(9.45, 7.85, "Architecture B — BiLSTM", fontsize=9, ha="center", fontweight="bold")
lstmB = box(ax, (7.2, 3.9), 4.7, 2.35,
            "BiLSTM, 2 layers, hidden=192,\nbidirectional, dropout=0.1\n(forward + backward pass over\nfull utterance, non-causal)",
            fc="#eef7ea")
headB = box(ax, (7.2, 2.5), 4.7, 0.65, "Linear head: 384 -> 192 (ReLU) -> 14, sigmoid")
outB = box(ax, (7.2, 1.1), 4.7, 0.8, "b_hat(t) in [0,1]^14\n(ARKit blendshapes, 60 fps)",
           fc="#fff3e0")

arrow(ax, inp, lstmB, "right", "top")
arrow(ax, lstmB, headB, "bottom", "top")
arrow(ax, headB, outB, "bottom", "top")

ax.set_title("Auxiliary audio-to-blendshape regression architectures (Section V-F)",
              fontsize=11, pad=10)
save(fig, "fig5_aux_architectures")

print("Wrote figures to", os.path.abspath(OUT))
