"""Figure: qualitative montage of the rendered VRM avatar across conditions.

Extracts a single, frame-accurate articulation instant from one held-out
utterance (chat00, t = 1.40 s) rendered under the four evaluation anchors
(AMP / RATE / TSYNC / MFA) and lays them side by side. The frames come from the
same 300-video render set scored by SyncNet in Section V-C, and make concrete
the *textureless, stylized* nature of the avatar — the property we argue drives
the SyncNet LSE-D/LSE-C discrepancy (Section V-C / VI-B).

Run from anywhere; needs ffmpeg on PATH and paper/eval_work/videos/*.mp4.
Output: paper/figures/fig_avatar_conditions.{png,pdf}
"""
import os
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(__file__)
VIDEOS = os.path.normpath(os.path.join(HERE, "..", "..", "eval_work", "videos"))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "figures"))

UTT = "chat00"
T = "1.40"                      # articulation instant (open-mouth frame)
CROP = (120, 30, 520, 390)      # (left, top, right, bottom) -> focus on the face
PANELS = [
    ("amp", "AMP\n(lower anchor)"),
    ("rate", "RATE\n(deployed)"),
    ("tsync", "TSYNC\n(ours)"),
    ("mfa", "MFA\n(oracle)"),
]


def grab(cond, dst):
    src = os.path.join(VIDEOS, f"{UTT}_{cond}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", T, "-i", src,
         "-frames:v", "1", dst],
        check=True,
    )
    return Image.open(dst).convert("L").crop(CROP)


def main():
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        imgs = [(grab(c, os.path.join(tmp, f"{c}.png")), lab) for c, lab in PANELS]

    fig, axes = plt.subplots(1, len(PANELS), figsize=(10.5, 3.5))
    for ax, (im, lab) in zip(axes, imgs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(lab, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#999999")
    fig.suptitle(
        f"Rendered VRM avatar, identical utterance and instant "
        f"({UTT}, t = {T} s), four conditions",
        fontsize=11, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig_avatar_conditions.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_avatar_conditions.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Wrote fig_avatar_conditions.{png,pdf}")


if __name__ == "__main__":
    main()
