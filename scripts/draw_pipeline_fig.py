#!/usr/bin/env python3
"""Draw Fig. 1: libeventgate reconstruction pipeline with HOLD and ARE gate."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("figures/imu_gated_evs_hold/fig_pipeline.png")

BOXES = [
    (0.04, 0.52, 0.14, 0.28, "HDF5 events\n+ IMU CSV"),
    (0.24, 0.52, 0.16, 0.28, "10 ms voxels\n5 bins, CUDA"),
    (0.46, 0.52, 0.16, 0.28, "FireNet\nTensorRT"),
    (0.68, 0.52, 0.14, 0.28, "8-bit recon\nORB count"),
    (0.86, 0.52, 0.11, 0.28, "recon.mp4\nkeypoints"),
    (0.24, 0.08, 0.16, 0.28, "Empty window\nHOLD\nskip infer,\nre-emit last frame"),
    (0.46, 0.08, 0.16, 0.28, r"ARE: $\|\omega\|<\delta$" "\nSTATIC\ndiscard $h_{out}$\nrestore $h_{in}$"),
]


def box(ax, x, y, w, h, text, fc="#e8eef4"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=1.1,
        edgecolor="#1f4e79",
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color="#1b1b1b")


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#333333",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, y, w, h, t in BOXES[:5]:
        box(ax, x, y, w, h, t)
    box(ax, *BOXES[5][:4], BOXES[5][4], fc="#fbe5d6")
    box(ax, *BOXES[6][:4], BOXES[6][4], fc="#e2efda")
    arrow(ax, 0.18, 0.66, 0.24, 0.66)
    arrow(ax, 0.40, 0.66, 0.46, 0.66)
    arrow(ax, 0.62, 0.66, 0.68, 0.66)
    arrow(ax, 0.82, 0.66, 0.86, 0.66)
    arrow(ax, 0.32, 0.36, 0.32, 0.52)
    arrow(ax, 0.54, 0.36, 0.54, 0.52)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
