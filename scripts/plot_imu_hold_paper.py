#!/usr/bin/env python3
"""Regenerate paper figures from session_1136 keypoints + imu.csv."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TMIN_US = 4593  # event_t_min_us from eventgate_usb/MANIFEST.json
DEFAULT_THRESH = 0.08  # rad/s; near this bag's median ||ω||


def load_imu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t, n = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            ts = (int(row["timestamp_us"]) - TMIN_US) * 1e-6
            gx, gy, gz = float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"])
            t.append(ts)
            n.append(math.sqrt(gx * gx + gy * gy + gz * gz))
    return np.array(t), np.array(n)


def load_kp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t, c = [], []
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(float(row[0]))
            c.append(int(float(row[1])))
    return np.array(t), np.array(c)


def shade_static(ax, imu_t, imu_n, th, label):
    static = imu_n < th
    i, first = 0, True
    while i < len(static):
        if static[i]:
            j = i
            while j < len(static) and static[j]:
                j += 1
            ax.axvspan(
                imu_t[i],
                imu_t[j - 1],
                color="#548235",
                alpha=0.15,
                label=label if first else None,
            )
            first = False
            i = j
        else:
            i += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--imu", default="eventgate_usb/imu.csv")
    p.add_argument("--keypoints", default="out/session_1136/keypoints.csv")
    p.add_argument("--keypoints-gated", default="out/session_1136_gated/keypoints.csv")
    p.add_argument("--out", default="figures/imu_gated_evs_hold")
    p.add_argument("--thresh", type=float, default=DEFAULT_THRESH)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imu_t, imu_n = load_imu(Path(args.imu))
    kp_t, kp_c = load_kp(Path(args.keypoints))
    th = args.thresh
    gated_path = Path(args.keypoints_gated)
    kp_g = load_kp(gated_path)[1] if gated_path.is_file() else None

    step = 10
    fig, ax = plt.subplots(
        2, 1, figsize=(8.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]}
    )
    ax[0].plot(
        kp_t[::step],
        kp_c[::step],
        color="#1f4e79",
        lw=0.8,
        label="ORB, gate off",
    )
    if kp_g is not None:
        ax[0].plot(
            kp_t[::step],
            kp_g[::step],
            color="#c45911",
            lw=0.8,
            label="ORB, ARE gate on",
        )
    ax[0].axvspan(168.14, 170.13, color="#c45911", alpha=0.18, label="Empty-window HOLD tail (2.0 s)")
    ax[0].set_ylabel("ORB keypoints")
    ax[0].set_ylim(0, 2100)
    ax[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    title = "Session 20260807_1136: FireNet, ungated vs ARE gate"
    if kp_g is None:
        title = "Session 20260807_1136: FireNet recon, IMU gate off"
    ax[0].set_title(title)

    ax[1].plot(imu_t, imu_n, color="#548235", lw=0.6, label=r"$\|\omega\|$ (rad/s)")
    ax[1].axhline(th, color="#c45911", ls="--", lw=1, label=rf"ARE thresh $\delta={th:.2f}$ rad/s")
    shade_static(ax[1], imu_t, imu_n, th, rf"STATIC ($\|\omega\|<{th:.2f}$)")
    ax[1].set_xlabel("Time (s, event domain)")
    ax[1].set_ylabel(r"$\|\omega\|$ (rad/s)")
    ax[1].set_ylim(0, 2.2)
    ax[1].legend(loc="upper right", fontsize=8, frameon=False)
    for a in ax:
        a.grid(True, alpha=0.3)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_keypoints_gyro.png", dpi=160)
    fig.savefig(out / "fig_keypoints_gyro.pdf")

    fig2, ax2 = plt.subplots(figsize=(5.2, 3.4))
    ax2.hist(imu_n, bins=60, color="#548235", alpha=0.85, range=(0, 1.5))
    ax2.axvline(
        th,
        color="#c45911",
        ls="--",
        label=rf"$\delta={th:.2f}$ ({100 * (imu_n < th).mean():.0f}% of samples)",
    )
    ax2.set_xlabel(r"$\|\omega\|$ (rad/s)")
    ax2.set_ylabel("IMU samples")
    ax2.legend(fontsize=8, frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(out / "fig_gyro_hist.png", dpi=160)
    print(f"wrote figures under {out}")


if __name__ == "__main__":
    main()
