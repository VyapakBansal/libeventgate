# Method: IMU-gated FireNet hold on a pure EVS

This is the pipeline as implemented in `src/pipeline.cpp` and `src/gate.cpp`.
Keep `IMU_Gated_EVS_Static_Hold.md` and `paper/main.tex` in lockstep with it.
Do not describe a detector the code does not run.

## Claim

A MEMS gyroscope is enough to decide that a pure event camera is still, and
that binary decision is enough to hold a FireNet reconstruction. The IMU is
not used for velocity, pose, SLAM, or ATE. Radar and DIRG are out of scope.

## Inputs

- Events: HDF5 structured array `(x, y, t_us, p)` from a Lucid Triton2 / Sony
  IMX637. No concurrent grayscale (not a DAVIS).
- IMU: CSV `timestamp_us, gyro_xyz, accel_xyz`. Sidecar rate on the current
  bag is about 24 Hz. Software-aligned into the event clock (not PTP).
- FireNet TensorRT engine with `h_in` / `h_out` exposed. Weights are not in
  git.

Time is partitioned into 10 ms windows. Events in window `k` become a 5-bin
voxel grid `V_k` on the GPU. FireNet maps `(V_k, h_{k-1}) -> (Ihat_k, h_k)`.
Float reconstructions are min-max stretched to 8-bit for ORB and MPEG-4.

## Three configurations (do not conflate)

1. **Ungated.** FireNet runs every nonempty window. Empty windows still HOLD.
2. **Freeze-h ablation** (`--ablate-freeze-h`, implies `--gate`). FireNet
   still runs during STATIC. `h_out` is discarded; `h_in` is restored. The
   emitted image is still `f(V_k, h_held)`, so the canvas fades. This is
   Table II only. Plain `--gate` is the latch, not this ablation.
3. **Fade-horizon latch (released default).** After 1 s of confirmed STATIC,
   skip `f` entirely and re-emit a stored 8-bit frame. Hidden state is not
   decoded from a frozen `h`; it is simply not updated. This is Section VII-D
   and Table IV.

## Detector

One-sample ARE on the nearest IMU sample to the window start:

    d_k = 1[ ||omega_k|| < delta ]

Default `delta = 0.08` rad/s on the current bag (near the median). This is
Skog ARE with window length 1, a threshold on gyro energy. Accelerometer
magnitude is not the gate. SHOE (accel *variance* fused with ARE) is not
implemented; the IMU is too slow. Adding `||a - g n-hat||` does not fix a
constant-velocity slider.

## Hysteresis

With `dt = 10` ms, `N_min = 100` (1 s) and `N_rel = 25` (0.25 s):

    s_k = s_{k-1}+1 if d_k=1 else 0
    r_k = 0 if d_k=1 or n_k=0 else r_{k-1}+1
    D_k = 1[s_k >= N_min]  or  (already held and r_k < N_rel)

Unconfirmed STATIC keeps inferring. Snapshot the latch candidate at STATIC
*onset* so the 1 s wait does not fill the buffer with dying frames. Apply the
snapshot only when STATIC lasts 1 s. Release after 0.25 s of MOVING windows
that contain events.

## Empty-window HOLD

If `n_k = 0` and a previous frame exists: skip voxelization and FireNet,
re-emit the last frame. Independent of the gyro. The configurable tail after
the last event in a file (2 s in the paper) is this path.

## Fade-horizon latch

Keep a rolling buffer of the last `L = 100` inferred `(Ihat_j, n_j)` pairs
(1 s). At STATIC onset:

    n* = max n_j in the buffer
    ell(k) = max { j in buffer : n_j >= n*/2 }

That is the most recent still-alive window, not the busiest (maybe smeared)
and not the first drop after the peak. If the buffer is empty, fall back to
the current 8-bit conversion of `h_frame`.

Lookback 1 s is the FireNet fade timescale (Scheerlinck WACV 2020 and the
ungated run), not a fit to ORB:

- 0.2 s: still in the deceleration tail, already gray.
- 1 s: skip that tail, heading still the stop pose.
- 2 s: can latch a canvas from a different heading.
- min-static 5 s: waits past the fade, freezes a dead frame.

Defaults: `--hold-lookback-s 1 --hold-min-static-s 1 --hold-release-s 0.25`.
With `--gate`, TEDG/CMDG/SRB are on by default (`--no-tedg`, `--no-cmdg`,
`--no-srb` to disable).

## Event-native extensions (released with `--gate`)

**TEDG.** When event rate falls for two consecutive windows before gyro static,
arm the latch candidate early:
\[
\gamma_k = \mathbf{1}[n_k \ge n_{\min} \wedge n_{k-1} \ge n_{\min} \wedge
n_k < \alpha n_{k-1} \wedge n_{k-1} < \alpha n_{k-2}]
\]
Default \(\alpha=0.5\), \(n_{\min}=100\). Does not override \(d_k\).

**CMDG.** Veto static when event center-of-mass drifts:
\[
\bar{\mathbf{c}}_k = \frac{1}{n_k}\sum_i (x_i,y_i),\quad
\Delta_k = \|\bar{\mathbf{c}}_k - \bar{\mathbf{c}}_{k-1}\|
\]
If \(\Delta_k > \tau_{\mathrm{com}}\) (default 3 px/window) while \(d_k=1\),
do not latch.

**SRB.** On release, blend held frame with live inference over \(N_b=10\) windows:
\[
\hat{I}_k^{\mathrm{out}} = (1-\beta_m)\hat{I}_\ell + \beta_m f(V_k,h_{k-1}),\quad
\beta_m = m/N_b.
\]

## Released update

    (Ihat_k, h_k) =
      (Ihat_{k-1}, h_{k-1})     if n_k = 0
      (Ihat_ell,   h_{k-1})     if D_k = 1   (skip f)
      f(V_k, h_{k-1})           otherwise

Stereo CLI (`--events-left`, `--events-right`, one IMU) applies the same
`D_k` sequentially. Not measured. One 6 GB GPU, one engine.

## Metrics

**Table II (freeze-h only).** ORB count, cap 2000, split STATIC/MOVING by
nearest IMU sample. MOVING should stay near ungated. STATIC should drop.
HOLD tail should be identical counts. ORB is a freeze detector on stretched
8-bit, not a tracker.

**Table IV (placeholders).** After each STATIC bout of at least 1 s, take
`I_ell` from the *latch* video at `t0+1` s. Never use the starved window
where `d_k` first flips as SSIM ground truth.

- SSIM of each method against `I_ell` over `[t0+1, t1]`
- Mean Farneback `||v||` between consecutive sampled frames
- KLT survival of Shi-Tomasi points from `t0+1` to `t1`, only for bouts >= 5 s
  (target on the next bag: four planned stops >= 10 s)

Script: `scripts/stability_metrics.py`. NIQE, BRISQUE, LPIPS are not used.

## Known limit

Gyro-only ARE is blind to translation with no rotation (slider, straight
road). That is a boundary of the claim, not a bug. A visual AND (gyro quiet
and low flow) or real SHOE on a faster IMU are the honest next detectors.
Magnitude-SHOE is not.
