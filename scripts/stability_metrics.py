#!/usr/bin/env python3
"""Temporal stability metrics for STATIC bouts.

SSIM is always against the latched canvas I_ell (lookback frame at t0+1s),
never against the starved window where ||ω|| first crosses δ.

After a new bag: same flags, same output names.
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

TMIN_DEFAULT = 4593


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"event_t_min_us": TMIN_DEFAULT, "session_id": "session"}
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


def static_bouts(imu_t, imu_n, th) -> list[tuple[float, float]]:
    out, i, s = [], 0, imu_n < th
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


def pick_stops(bouts, n_stops, min_stop_s):
    long = [b for b in bouts if (b[1] - b[0]) >= min_stop_s]
    long.sort(key=lambda b: b[1] - b[0], reverse=True)
    chosen = sorted(long[:n_stops], key=lambda b: b[0])
    return chosen


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a = ((a - mu_a) ** 2).mean()
    sig_b = ((b - mu_b) ** 2).mean()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (sig_a + sig_b + c2)))


def grab(cap: cv2.VideoCapture, fps: float, t: float) -> np.ndarray | None:
    idx = int(round(t * fps))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(max(idx, 0), max(n - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        return None
    if fr.ndim == 3:
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    return fr


def flow_mag(a, b) -> float:
    fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
    return float(np.mean(mag))


def klt_survival(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return float("nan")
    pts = cv2.goodFeaturesToTrack(frames[0], maxCorners=200, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) == 0:
        return float("nan")
    n0 = len(pts)
    prev, cur_pts = frames[0], pts
    alive = n0
    for fr in frames[1:]:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, fr, cur_pts, None)
        if nxt is None or st is None:
            alive = 0
            break
        good = st.reshape(-1) == 1
        alive = int(good.sum())
        if alive == 0:
            break
        cur_pts = nxt[good].reshape(-1, 1, 2)
        prev = fr
    return 100.0 * alive / n0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="eventgate_usb/MANIFEST.json")
    p.add_argument("--imu", default="eventgate_usb/imu.csv")
    p.add_argument("--video-ungated", default="out/session_1136/recon.mp4")
    p.add_argument("--video-gated", default="out/session_1136_gated/recon.mp4")
    p.add_argument("--video-lookback", default="out/session_1136_enhanced/recon.mp4")
    p.add_argument("--out", default="figures/imu_gated_evs_hold")
    p.add_argument("--thresh", type=float, default=0.08)
    p.add_argument("--n-stops", type=int, default=4)
    p.add_argument("--min-stop-s", type=float, default=1.0)
    p.add_argument("--step", type=int, default=10, help="sample every N 10 ms windows")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(Path(args.manifest))
    tmin = int(man.get("event_t_min_us", TMIN_DEFAULT))
    imu_t, imu_n = load_imu(Path(args.imu), tmin)
    stops = pick_stops(static_bouts(imu_t, imu_n, args.thresh), args.n_stops, args.min_stop_s)
    names = {
        "ungated": Path(args.video_ungated),
        "freeze-h": Path(args.video_gated),
        "latch": Path(args.video_lookback),
    }
    caps = {}
    fps = 100.0
    for k, path in names.items():
        if path.is_file():
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 100.0
            caps[k] = cap
    if "latch" not in caps:
        print("WARN: no lookback video; cannot form I_ell")
        return

    rows = []
    ssim_series = {k: [] for k in caps}
    flow_series = {k: [] for k in caps}
    t_series = []

    for si, (t0, t1) in enumerate(stops, start=1):
        t_ref = min(t0 + 1.0, t1)  # after min-static, latched canvas
        i_ell = grab(caps["latch"], fps, t_ref)
        if i_ell is None:
            continue
        rec = {"stop": si, "t0": t0, "t1": t1, "t_ref": t_ref, "duration_s": t1 - t0}
        times = np.arange(t_ref, t1, args.step * 0.01)
        for name, cap in caps.items():
            ssims, flows = [], []
            prev = None
            klt_frames = []
            for t in times:
                fr = grab(cap, fps, t)
                if fr is None:
                    continue
                ssims.append(ssim_gray(fr, i_ell))
                if prev is not None:
                    flows.append(flow_mag(prev, fr))
                prev = fr
                klt_frames.append(fr)
            rec[f"ssim_{name}"] = float(np.mean(ssims)) if ssims else None
            rec[f"flow_{name}"] = float(np.mean(flows)) if flows else None
            rec[f"klt_{name}"] = klt_survival(klt_frames) if (t1 - t0) >= 5.0 else None
            ssim_series[name].append(rec[f"ssim_{name}"])
            flow_series[name].append(rec[f"flow_{name}"])
        t_series.append(si)
        rows.append(rec)
        print(f"stop {si} {t0:.1f}-{t1:.1f}s  " +
              "  ".join(f"SSIM {n}={rec.get(f'ssim_{n}'):.3f}" if rec.get(f"ssim_{n}") is not None else ""
                        for n in caps))

    (out / "stability_metrics.json").write_text(json.dumps({"stops": rows}, indent=2))

    def barplot(series, ylabel, fname, ylim=None):
        x = np.arange(len(t_series))
        w = 0.25
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for i, (name, color) in enumerate(
            (("ungated", "#1f4e79"), ("freeze-h", "#7f7f7f"), ("latch", "#c45911"))
        ):
            if name not in series:
                continue
            y = [np.nan if v is None else v for v in series[name]]
            ax.bar(x + (i - 1) * w, y, w, label=name, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Stop {i}" for i in t_series])
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=200)
        fig.savefig(out / fname.replace(".png", ".pdf"))
        plt.close(fig)

    barplot(ssim_series, r"Mean SSIM vs $\hat{I}_\ell$", "fig_ssim_stops.png", (0, 1.05))
    barplot(flow_series, r"Mean Farneback $\|\mathbf{v}\|$ (px)", "fig_flow_stops.png")
    klt_series = {k: [r.get(f"klt_{k}") for r in rows] for k in caps}
    barplot(klt_series, "KLT survival at stop end (%)", "fig_klt_stops.png", (0, 105))
    for cap in caps.values():
        cap.release()
    print(f"wrote {out}/stability_metrics.json")


if __name__ == "__main__":
    main()
