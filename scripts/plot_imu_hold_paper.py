#!/usr/bin/env python3
"""Regenerate paper figures from keypoints CSVs, IMU sidecar, and recon videos.

Bag-agnostic: times, HOLD tail, and the zoom window are derived from
eventgate_usb/MANIFEST.json plus the CSVs. After a new recording of similar
length, point --imu/--keypoints/--video-* at the new outputs and re-run.
Do not edit the PNGs by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_THRESH = 0.08


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"session_id": "session", "event_t_min_us": 0}
    return json.loads(path.read_text())


def load_imu(path: Path, tmin_us: int) -> tuple[np.ndarray, np.ndarray]:
    t, n = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            ts = (int(row["timestamp_us"]) - tmin_us) * 1e-6
            gx, gy, gz = float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"])
            t.append(ts)
            n.append(math.sqrt(gx * gx + gy * gy + gz * gz))
    return np.asarray(t), np.asarray(n)


def load_kp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t, c = [], []
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(float(row[0]))
            c.append(int(float(row[1])))
    return np.asarray(t), np.asarray(c)


def trailing_identical(kp_t: np.ndarray, kp_c: np.ndarray) -> tuple[float, float, int, int]:
    """Start/end time, count, and window count of the trailing identical ORB run."""
    if len(kp_c) == 0:
        return 0.0, 0.0, 0, 0
    i = len(kp_c) - 1
    while i > 0 and kp_c[i] == kp_c[-1]:
        i -= 1
    i += 1
    return float(kp_t[i]), float(kp_t[-1]), int(kp_c[-1]), int(len(kp_c) - i)


def static_bouts(imu_t: np.ndarray, imu_n: np.ndarray, th: float) -> list[tuple[float, float]]:
    out = []
    i = 0
    s = imu_n < th
    while i < len(s):
        if s[i]:
            j = i
            while j < len(s) and s[j]:
                j += 1
            out.append((float(imu_t[i]), float(imu_t[j - 1])))
            i = j
        else:
            i += 1
    return out


def nearest_imu_mask(kp_t: np.ndarray, imu_t: np.ndarray, imu_n: np.ndarray, th: float):
    static = np.full(len(kp_t), np.nan)
    if len(imu_t) == 0:
        return static
    j = 0
    for i, t in enumerate(kp_t):
        while j + 1 < len(imu_t) and abs(imu_t[j + 1] - t) <= abs(imu_t[j] - t):
            j += 1
        if abs(imu_t[j] - t) > 0.1:
            continue
        static[i] = 1.0 if imu_n[j] < th else 0.0
    return static


def summarize(name: str, kp_c: np.ndarray, static: np.ndarray, tail_n: int) -> dict:
    m_s = static == 1
    m_m = static == 0
    tail = kp_c[-tail_n:] if tail_n > 0 else kp_c[-1:]
    rec = {
        "name": name,
        "n": int(len(kp_c)),
        "mean": float(kp_c.mean()),
        "median": float(np.median(kp_c)),
        "static_n": int(m_s.sum()),
        "static_mean": float(kp_c[m_s].mean()) if m_s.any() else None,
        "static_median": float(np.median(kp_c[m_s])) if m_s.any() else None,
        "moving_n": int(m_m.sum()),
        "moving_mean": float(kp_c[m_m].mean()) if m_m.any() else None,
        "moving_median": float(np.median(kp_c[m_m])) if m_m.any() else None,
        "hold_tail_count": int(tail[0]),
        "hold_tail_identical": bool(np.all(tail == tail[0])),
    }
    print(f"\n=== {name} ===")
    print(f"  n={rec['n']} mean={rec['mean']:.1f} median={rec['median']:.1f}")
    if rec["static_n"]:
        print(
            f"  STATIC n={rec['static_n']} mean={rec['static_mean']:.1f} "
            f"median={rec['static_median']:.1f}"
        )
    if rec["moving_n"]:
        print(
            f"  MOVING n={rec['moving_n']} mean={rec['moving_mean']:.1f} "
            f"median={rec['moving_median']:.1f}"
        )
    print(
        f"  HOLD tail n={tail_n} count={rec['hold_tail_count']} "
        f"identical={rec['hold_tail_identical']}"
    )
    return rec


def shade_static(ax, imu_t, imu_n, th, label):
    first = True
    for t0, t1 in static_bouts(imu_t, imu_n, th):
        ax.axvspan(t0, t1, color="#548235", alpha=0.15, label=label if first else None)
        first = False


def grab_frame(video: Path, t_s: float, window_s: float = 0.01) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"WARN: cannot open {video}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or (1.0 / window_s)
    idx = int(round(t_s * fps))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(max(idx, 0), max(n - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"WARN: failed frame t={t_s} in {video}")
        return None
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def save_gray(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def pick_motion_time(imu_t: np.ndarray, imu_n: np.ndarray, th: float) -> float:
    """Peak ||ω|| among MOVING samples in the first third of the bag."""
    if len(imu_t) == 0:
        return 0.0
    t_cut = imu_t[0] + (imu_t[-1] - imu_t[0]) / 3.0
    m = (imu_t <= t_cut) & (imu_n >= th)
    if not m.any():
        m = imu_n >= th
    if not m.any():
        return float(imu_t[len(imu_t) // 4])
    idx = np.argmax(imu_n[m])
    return float(imu_t[m][idx])


def pick_stops(
    bouts: list[tuple[float, float]], n_stops: int, min_stop_s: float
) -> list[tuple[float, float]]:
    """N longest STATIC bouts of at least min_stop_s, then in time order."""
    long = [b for b in bouts if (b[1] - b[0]) >= min_stop_s]
    long.sort(key=lambda b: b[1] - b[0], reverse=True)
    chosen = long[: max(n_stops, 0)]
    chosen.sort(key=lambda b: b[0])
    return chosen


def show_gray(ax, img: np.ndarray | None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if img is None:
        ax.set_facecolor("#ddd")
        return
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)


def plateaus(kp_t: np.ndarray, kp_c: np.ndarray, min_s: float = 0.5) -> list[dict]:
    out = []
    i = 0
    while i < len(kp_c):
        k = i + 1
        while k < len(kp_c) and kp_c[k] == kp_c[i]:
            k += 1
        dur = float(kp_t[k - 1] - kp_t[i])
        if dur >= min_s:
            out.append(
                {
                    "t0": float(kp_t[i]),
                    "t1": float(kp_t[k - 1]),
                    "n_windows": int(k - i),
                    "orb": int(kp_c[i]),
                }
            )
        i = k
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="eventgate_usb/MANIFEST.json")
    p.add_argument("--imu", default="eventgate_usb/imu.csv")
    p.add_argument("--keypoints", default="out/session_1136/keypoints.csv")
    p.add_argument("--keypoints-gated", default="out/session_1136_gated/keypoints.csv")
    p.add_argument("--keypoints-lookback", default="out/session_1136_enhanced/keypoints.csv")
    p.add_argument("--video-ungated", default="out/session_1136/recon.mp4")
    p.add_argument("--video-gated", default="out/session_1136_gated/recon.mp4")
    p.add_argument("--video-lookback", default="out/session_1136_enhanced/recon.mp4")
    p.add_argument("--out", default="figures/imu_gated_evs_hold")
    p.add_argument("--thresh", type=float, default=DEFAULT_THRESH)
    p.add_argument("--n-stops", type=int, default=4, help="Longest STATIC bouts to still/zoom")
    p.add_argument("--min-stop-s", type=float, default=1.0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    man = load_manifest(Path(args.manifest))
    tmin_us = int(man.get("event_t_min_us", 0))
    session = str(man.get("session_id", "session"))
    th = args.thresh

    imu_t, imu_n = load_imu(Path(args.imu), tmin_us)
    kp_t, kp_c = load_kp(Path(args.keypoints))
    static = nearest_imu_mask(kp_t, imu_t, imu_n, th)
    hold0, hold1, hold_c, hold_n = trailing_identical(kp_t, kp_c)
    bouts = static_bouts(imu_t, imu_n, th)
    longest = max(bouts, key=lambda b: b[1] - b[0]) if bouts else (0.0, 1.0)
    stops = pick_stops(bouts, args.n_stops, args.min_stop_s)
    if not stops:
        stops = [longest]
    zoom0 = max(0.0, longest[0] - 4.0)
    zoom1 = longest[1] + 4.0
    t_motion = pick_motion_time(imu_t, imu_n, th)
    t_static = 0.5 * (longest[0] + longest[1])
    t_hold = 0.5 * (hold0 + hold1)
    stop_times = [0.5 * (a + b) for a, b in stops]

    gated_path = Path(args.keypoints_gated)
    look_path = Path(args.keypoints_lookback)
    kp_g = load_kp(gated_path)[1] if gated_path.is_file() else None
    kp_l = load_kp(look_path)[1] if look_path.is_file() else None

    stats = {
        "session_id": session,
        "delta": th,
        "hold_tail": {"t0": hold0, "t1": hold1, "count": hold_c, "n_windows": hold_n},
        "longest_static": {"t0": longest[0], "t1": longest[1]},
        "still_times": {"motion": t_motion, "static": t_static, "hold": t_hold},
        "stops": [{"t0": a, "t1": b, "mid": 0.5 * (a + b)} for a, b in stops],
        "runs": [],
    }
    series = [("ungated", kp_c)]
    if kp_g is not None:
        series.append(("freeze-h (ablation)", kp_g))
    if kp_l is not None:
        series.append(("fade-horizon latch", kp_l))
        stats["latch_plateaus"] = plateaus(kp_t, kp_l)
    for name, arr in series:
        stats["runs"].append(summarize(name, arr, static, hold_n))

    step = max(1, len(kp_t) // 1700)
    fig, ax = plt.subplots(
        2, 1, figsize=(8.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]}
    )
    ax[0].plot(kp_t[::step], kp_c[::step], color="#1f4e79", lw=0.8, label="ORB, gate off")
    if kp_g is not None:
        ax[0].plot(
            kp_t[::step],
            kp_g[::step],
            color="#7f7f7f",
            lw=0.7,
            ls=":",
            label="ORB, freeze-$h$ ablation",
        )
    if kp_l is not None:
        ax[0].plot(
            kp_t[::step],
            kp_l[::step],
            color="#c45911",
            lw=0.9,
            label="ORB, fade-horizon latch",
        )
    ax[0].axvspan(hold0, hold1, color="#c45911", alpha=0.18, label="Empty-window HOLD tail")
    ax[0].set_ylabel("ORB keypoints")
    ax[0].set_ylim(0, 2100)
    ax[0].legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    ax[0].set_title(f"{session}: FireNet, ungated vs fade-horizon latch")

    ax[1].plot(imu_t, imu_n, color="#548235", lw=0.6, label=r"$\|\omega\|$ (rad/s)")
    ax[1].axhline(th, color="#c45911", ls="--", lw=1, label=rf"ARE thresh $\delta={th:.2f}$ rad/s")
    shade_static(ax[1], imu_t, imu_n, th, rf"STATIC ($\|\omega\|<{th:.2f}$)")
    ax[1].set_xlabel("Time (s, event domain)")
    ax[1].set_ylabel(r"$\|\omega\|$ (rad/s)")
    ax[1].set_ylim(0, max(2.2, float(np.percentile(imu_n, 99)) if len(imu_n) else 2.2))
    ax[1].legend(loc="upper right", fontsize=8, frameon=False)
    for a in ax:
        a.grid(True, alpha=0.3)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_keypoints_gyro.png", dpi=200)
    fig.savefig(out / "fig_keypoints_gyro.pdf")
    plt.close(fig)

    figz, axz = plt.subplots(
        2, 1, figsize=(8.4, 4.8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]}
    )
    m = (kp_t >= zoom0) & (kp_t <= zoom1)
    axz[0].plot(kp_t[m], kp_c[m], color="#1f4e79", lw=1.0, label="ORB, gate off")
    if kp_g is not None:
        axz[0].plot(kp_t[m], kp_g[m], color="#7f7f7f", lw=0.9, ls=":", label="ORB, freeze-$h$")
    if kp_l is not None:
        axz[0].plot(kp_t[m], kp_l[m], color="#c45911", lw=1.1, label="ORB, fade-horizon latch")
    axz[0].set_ylabel("ORB keypoints")
    axz[0].set_ylim(0, 2100)
    axz[0].legend(loc="upper right", fontsize=8, frameon=False)
    axz[0].set_title(rf"Longest STATIC bout ($t={longest[0]:.1f}$--${longest[1]:.1f}$ s)")
    mi = (imu_t >= zoom0) & (imu_t <= zoom1)
    axz[1].plot(imu_t[mi], imu_n[mi], color="#548235", lw=0.8, label=r"$\|\omega\|$")
    axz[1].axhline(th, color="#c45911", ls="--", lw=1, label=rf"$\delta={th:.2f}$")
    shade_static(axz[1], imu_t[mi], imu_n[mi], th, "STATIC")
    axz[1].set_xlabel("Time (s, event domain)")
    axz[1].set_ylabel(r"$\|\omega\|$ (rad/s)")
    ymax = float(np.max(imu_n[mi])) if mi.any() else 0.6
    axz[1].set_ylim(0, max(0.6, 1.2 * ymax))
    axz[1].legend(loc="upper right", fontsize=8, frameon=False)
    for a in axz:
        a.grid(True, alpha=0.3)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    figz.tight_layout()
    figz.savefig(out / "fig_keypoints_zoom_static.png", dpi=200)
    figz.savefig(out / "fig_keypoints_zoom_static.pdf")
    plt.close(figz)

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
    fig2.savefig(out / "fig_gyro_hist.png", dpi=200)
    fig2.savefig(out / "fig_gyro_hist.pdf")
    plt.close(fig2)

    v_u = Path(args.video_ungated)
    v_g = Path(args.video_gated)
    v_l = Path(args.video_lookback)
    videos = {"ungated": v_u, "gated": v_g, "latch": v_l}

    stills = {
        "still_motion.png": t_motion,
        "still_static.png": t_static,
        "still_hold.png": t_hold,
        "still_hold_start.png": hold0,
        "still_hold_end.png": hold1,
    }
    for name, t in stills.items():
        for tag, vid in videos.items():
            if not vid.is_file():
                continue
            fr = grab_frame(vid, t)
            if fr is None:
                continue
            fname = name if tag == "ungated" else name.replace(".png", f"_{tag}.png")
            save_gray(out / fname, fr)

    # 20 stills: 4 stops x 3 methods + motion x 3 + HOLD tail x 3 + HOLD start/end
    for i, t in enumerate(stop_times, start=1):
        for tag, vid in videos.items():
            if not vid.is_file():
                continue
            fr = grab_frame(vid, t)
            if fr is not None:
                save_gray(out / f"stop{i}_{tag}.png", fr)
    for tag, vid in videos.items():
        if not vid.is_file():
            continue
        fr = grab_frame(vid, t_motion)
        if fr is not None:
            save_gray(out / f"motion_{tag}.png", fr)
        fr = grab_frame(vid, t_hold)
        if fr is not None:
            save_gray(out / f"hold_{tag}.png", fr)

    def panel_grid(path_stem: str, rows: list[tuple[str, Path]], cols: list[tuple[str, float]], w: float, h: float):
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(w, h))
        if len(rows) == 1:
            axes = np.array([axes])
        if len(cols) == 1:
            axes = axes.reshape(-1, 1)
        for r, (rlab, vid) in enumerate(rows):
            for c, (clab, t) in enumerate(cols):
                axc = axes[r, c]
                show_gray(axc, grab_frame(vid, t) if vid.is_file() else None)
                if r == 0:
                    axc.set_title(clab, fontsize=8)
                if c == 0:
                    axc.set_ylabel(rlab, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{path_stem}.png", dpi=200)
        fig.savefig(out / f"{path_stem}.pdf")
        plt.close(fig)

    stop_cols = [
        (rf"Stop {i} ($t={t:.1f}$ s)", t) for i, t in enumerate(stop_times, start=1)
    ]
    method_rows = [(lab, vid) for lab, vid in (("Ungated", v_u), ("Freeze-$h$", v_g), ("Latch", v_l))]
    if stop_cols:
        panel_grid("fig_stills_stops", method_rows, stop_cols, 10.4, 7.2)
        panel_grid(
            "fig_stills_ungated_vs_latch",
            [("Ungated", v_u), ("Latch", v_l)],
            stop_cols,
            10.4, 5.0,
        )
    panel_grid(
        "fig_stills_motion",
        method_rows,
        [(rf"Motion ($t={t_motion:.1f}$ s)", t_motion)],
        4.2, 7.0,
    )
    panel_grid(
        "fig_stills_hold",
        method_rows,
        [(rf"HOLD tail ($t={t_hold:.1f}$ s)", t_hold)],
        4.2, 7.0,
    )

    # Four stop zooms (ORB + gyro), one composite.
    nstop = max(len(stops), 1)
    figzs, axzs = plt.subplots(nstop, 2, figsize=(8.6, 2.2 * nstop), squeeze=False)
    for i, (a, b) in enumerate(stops):
        z0, z1 = max(0.0, a - 2.0), b + 2.0
        m = (kp_t >= z0) & (kp_t <= z1)
        axzs[i, 0].plot(kp_t[m], kp_c[m], color="#1f4e79", lw=0.8, label="ungated")
        if kp_g is not None:
            axzs[i, 0].plot(kp_t[m], kp_g[m], color="#7f7f7f", lw=0.7, ls=":", label="freeze-$h$")
        if kp_l is not None:
            axzs[i, 0].plot(kp_t[m], kp_l[m], color="#c45911", lw=0.9, label="latch")
        axzs[i, 0].set_ylabel("ORB")
        axzs[i, 0].set_ylim(0, 2100)
        axzs[i, 0].set_title(rf"Stop {i+1}: $t={a:.1f}$--${b:.1f}$ s", fontsize=9)
        if i == 0:
            axzs[i, 0].legend(fontsize=6, frameon=False, loc="upper right")
        mi = (imu_t >= z0) & (imu_t <= z1)
        axzs[i, 1].plot(imu_t[mi], imu_n[mi], color="#548235", lw=0.7)
        axzs[i, 1].axhline(th, color="#c45911", ls="--", lw=0.8)
        shade_static(axzs[i, 1], imu_t[mi], imu_n[mi], th, None)
        axzs[i, 1].set_ylabel(r"$\|\omega\|$")
        axzs[i, 1].set_ylim(0, max(0.5, 1.2 * float(np.max(imu_n[mi])) if mi.any() else 0.5))
        for axx in axzs[i]:
            axx.grid(True, alpha=0.3)
            axx.spines["top"].set_visible(False)
            axx.spines["right"].set_visible(False)
        axzs[i, 0].set_xlabel("t (s)")
        axzs[i, 1].set_xlabel("t (s)")
        figzi, axzi = plt.subplots(
            2, 1, figsize=(6.4, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]}
        )
        axzi[0].plot(kp_t[m], kp_c[m], color="#1f4e79", lw=0.9, label="ungated")
        if kp_g is not None:
            axzi[0].plot(kp_t[m], kp_g[m], color="#7f7f7f", lw=0.8, ls=":", label="freeze-$h$")
        if kp_l is not None:
            axzi[0].plot(kp_t[m], kp_l[m], color="#c45911", lw=1.0, label="latch")
        axzi[0].set_ylabel("ORB")
        axzi[0].set_ylim(0, 2100)
        axzi[0].legend(fontsize=7, frameon=False)
        axzi[0].set_title(rf"Stop {i+1} ($t={a:.1f}$--${b:.1f}$ s)")
        axzi[1].plot(imu_t[mi], imu_n[mi], color="#548235", lw=0.8)
        axzi[1].axhline(th, color="#c45911", ls="--", lw=1)
        shade_static(axzi[1], imu_t[mi], imu_n[mi], th, None)
        axzi[1].set_xlabel("Time (s)")
        axzi[1].set_ylabel(r"$\|\omega\|$")
        for axx in axzi:
            axx.grid(True, alpha=0.3)
            axx.spines["top"].set_visible(False)
            axx.spines["right"].set_visible(False)
        figzi.tight_layout()
        figzi.savefig(out / f"fig_zoom_stop{i+1}.png", dpi=200)
        figzi.savefig(out / f"fig_zoom_stop{i+1}.pdf")
        plt.close(figzi)
    figzs.tight_layout()
    figzs.savefig(out / "fig_zoom_stops.png", dpi=200)
    figzs.savefig(out / "fig_zoom_stops.pdf")
    plt.close(figzs)

    if v_u.is_file() and v_l.is_file():
        labels = [
            (t_motion, "Motion"),
            (t_static, "Long STATIC"),
            (t_hold, "HOLD tail"),
        ]
        figc, axes = plt.subplots(2, 3, figsize=(9.6, 5.4))
        for col, (t, lab) in enumerate(labels):
            uimg = grab_frame(v_u, t)
            limg = grab_frame(v_l, t)
            for row, img, tag in ((0, uimg, "Ungated"), (1, limg, "Latch")):
                axc = axes[row, col]
                show_gray(axc, img)
                if row == 0:
                    axc.set_title(rf"({chr(ord('a')+col)}) $t={t:.1f}$ s, {lab}", fontsize=9)
                if col == 0:
                    axc.set_ylabel(tag, fontsize=9)
        figc.tight_layout()
        figc.savefig(out / "fig_stills_compare.png", dpi=200)
        figc.savefig(out / "fig_stills_compare.pdf")
        plt.close(figc)

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nstops ({len(stops)}): " + ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in stops))
    print(f"still times motion={t_motion:.2f}s static={t_static:.2f}s hold={t_hold:.2f}s")
    print(f"HOLD tail {hold0:.2f}-{hold1:.2f}s count={hold_c} n={hold_n}")
    print(f"wrote figures under {out}")


if __name__ == "__main__":
    main()
