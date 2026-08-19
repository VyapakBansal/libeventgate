# Innovation memo (agent output)

Eight mechanisms past the current fade-horizon latch. Implement in
`src/pipeline.cpp` unless noted. Paper claim stays gyro-sufficient for hold;
event-native cues below are **extensions**, not replacements, unless measured
as such.

---

## 1. Temporal Event-Decay Gate (TEDG)

**Claim:** Detect static onset from a sudden drop in per-window event rate,
not only gyro silence.

**Why only here:** Three clocks: 10 ms windows, ~24 Hz IMU (~80 ms chatter),
1–2 s FireNet fade. `n_k` is already in the main loop; IMU is not.

**Update rule:** Before hysteresis, compute a decay bit from consecutive counts:

```cpp
bool decay = (n >= 2 && n_prev >= 2 &&
              n_k < alpha * n_prev && n_prev < alpha * n_prev2);
// Option A (early latch): at decay onset, snapshot latch candidate even if
//   gyro not yet static (do not skip infer until D_k).
// Option B (fuse): d_k = d_k_imu || decay;  // risky: forces static while moving
```

Prefer **Option A**: trigger `static_candidate = latch_hold_frame(...)` on
decay, keep gyro for `D_k`. Do not put this in `latch_hold_frame`; it belongs
in the main loop next to `want_static` (~line 236 in `pipeline.cpp`).

**Failure mode:** Brief occlusion, motion blur, or low-texture motion looks
like decay. Never override gyro to static during sustained rotation.

**Tomorrow test:** Plot `n_k`, `||omega||`, hold bit on deceleration into
each planned stop. Latch snapshot should land before the gray tail, without
false holds during cruise motion.

**Cost:** 3–5 h (α tune on pilot bag, then 240 s bag).

---

## 2. Dual Hidden-State Hold (DHSH)

**Claim:** Separate live `h` and held `h_hold` so image latch and ConvGRU
memory follow different release laws.

**Why only here:** Released path skips `f` but leaves `h` at last live update;
latched `u8` is a different object (`pipeline.cpp` hold branch vs infer branch).

**Update rule:**

```cpp
if (static_streak == 0 && want_static) { h_hold = copy(h); }
if (D_k == 1) {
  emit I_latch;
  // choose one: freeze h, or slow-decay h_hold with periodic empty f
} else if (just_released) {
  h = h_hold;  // or blend h with h_hold over N windows
}
```

**Failure mode:** VRAM doubles for state tensors; wrong merge on release
smears motion. Highest integration risk.

**Tomorrow test:** After >=10 s stop, count windows until ORB on release
matches ungated motion baseline. Compare with current cliff release.

**Cost:** 20–30 h. Defer until single-camera latch is paper-stable.

---

## 3. Voxel Entropy Gate (VEG)

**Claim:** Spatial/polarity entropy of the 5-bin voxel confirms “featureless
static” vs structured motion residue.

**Why only here:** GPU voxel `V_k` exists every infer window; IMU has no
spatial structure.

**Update rule:**

```cpp
float H = entropy(V_k);  // over bins x space or per-bin then mean
if (d_k_imu && H > H_thresh)  confirm toward D_k;
if (H <= H_low && n_k > n_min)  veto D_k;  // edges still present
```

**Failure mode:** Sparse scenes: high H while moving. Textured wall at rest:
low H, delayed hold.

**Tomorrow test:** Log H vs ORB during four stops. Veto should fire on slider
segments if combined with CMDG (below).

**Cost:** 5–8 h (host-side from staging or small D2H slice).

---

## 4. Center-of-Mass Motion Gate (CMDG)

**Claim:** Event COM drift vetoes static when gyro is quiet but the sensor
translates (slider).

**Why only here:** `(x_i, y_i)` per window in staging; pure INS cannot see
2D shift without events.

**Update rule:** Veto, do not assert static:

```cpp
float xc = sum(x)/n_k, yc = sum(y)/n_k;
if (n_k >= n_min && hypot(xc-xc_prev, yc-yc_prev) > com_thresh)
  want_static = false;  // or block D_k latch
```

**Failure mode:** Symmetric motion, vibration, hot-pixel jitter. Require
`n_k >= n_min` and optional median over 3 windows.

**Tomorrow test:** Smooth slider with `||omega|| < delta`. CMDG should keep
`D_k=0`. True stop: COM stable, gyro static, latch fires.

**Cost:** 4–6 h. **Best fix for the paper’s stated false-STATIC boundary.**

---

## 5. Soft Release Blending (SRB)

**Claim:** Blend latched frame into first live frames on release instead of
one-step cliff.

**Why only here:** Hold emits stored `u8`; release immediately runs `f(V_k,h)`
with stale `h`.

**Update rule:**

```cpp
if (release_phase && step < N_blend) {
  I_out = (1-alpha[step]) * I_latch + alpha[step] * I_new;
  ++step;
} else { I_out = I_new; }
```

**Failure mode:** Ghosting if α too slow; delayed sharp motion recovery.

**Tomorrow test:** KLT / visual on first 5 frames after release vs current
cliff. Pilot Table IV KLT already drops on ungated mid-stop; check release
tail specifically.

**Cost:** 6–8 h (output path only, low risk).

---

## 6. Hold-Frame Quality Check (HFQC)

**Claim:** Refuse to latch if the candidate frame is noise-dominated.

**Why only here:** `static_candidate` is already chosen at onset; ORB on
stretched 8-bit is cheap in-process.

**Update rule:**

```cpp
if (static_confirmed && orb(static_candidate) < Q_min) {
  do not enter hold; keep inferring or wait for next decay onset;
}
```

**Failure mode:** Legitimate low-texture stops never latch. Tune `Q_min` from
MOVING baseline on new bag.

**Tomorrow test:** Log ORB of `static_candidate` vs committed latch. No-hold
only on empty tail / gray canvas.

**Cost:** 3–5 h. Overlaps HFQC with existing ORB pipeline.

---

## 7. Intermittent Hidden Decay (IHD)

**Claim:** During image hold, periodically run `f(empty_voxel, h)` to decay
`h` while still emitting `I_latch`.

**Why only here:** ConvGRU forget dynamics on near-empty input; image and
state decouple.

**Update rule:**

```cpp
if (D_k && static_counter % N_decay == 0) {
  f(empty_voxel, h);  // update h only
  u8 = I_latch;       // do not emit f output
}
```

**Failure mode:** Extra TRT cost; weak forget gate → no benefit. Opposite of
freeze-h ablation goal for the *image*, but may fix release smear.

**Tomorrow test:** Compare post-release ORB recovery time with/without IHD on
one long stop.

**Cost:** 6–8 h. Experimental.

---

## 8. Multi-Window Rate Gate (MWRG)

**Claim:** Short vs long EMA of `n_k` confirms static only when both collapse.

**Why only here:** Fixed 10 ms windows + streaming counts; multi-scale without
new sensors.

**Update rule:**

```cpp
E_short = ema(n_k, tau=0.1s);
E_long  = ema(n_k, tau=1.0s);
decay_confirmed = (E_short < beta * E_long) && (E_long < n_floor);
// Fuse with TEDG or gyro: use for latch index timing, not solo D_k.
```

**Failure mode:** Latency: hold starts late. Redundant with TEDG if both ship;
pick one first.

**Tomorrow test:** Plot E_short, E_long, gyro static on deceleration. Fewer
spurious early snapshots than raw TEDG.

**Cost:** 4–6 h.

---

## Implementation priority (agent ranking)

| Rank | Idea | Notes |
|------|------|--------|
| 1 | TEDG | Low cost; fixes time-warp if used for **early snapshot**, not gyro override |
| 2 | CMDG | Slider blind spot; veto semantics |
| 3 | SRB | Visible release quality |
| 4 | VEG | Refine latch when n*/2 lies |
| 5 | HFQC | Quick ORB gate on candidate |
| 6 | MWRG | Hysteresis on counts; merge with TEDG later |
| 7 | IHD | Split-brain experiment |
| 8 | DHSH | High cost / VRAM |

## Revised priority for tomorrow (author + codebase)

1. **CMDG** first: directly tests the paper’s false-STATIC boundary; 4–6 h;
   veto-only, honest claim extension.
2. **TEDG Option A** second: early `static_candidate` on decay; do **not**
   force `d_k=1` against gyro on day one.
3. **SRB** third if release smear is visible on the new bag; else **HFQC**.

Do not build DHSH until the 240 s bag and three-way runs are done. Do not
ship TEDG + MWRG together on day one (duplicate logic).

## Do not build (yet)

- **DHSH** until latch + metrics stable on planned-stop bag.
- **TEDG Option B** (gyro override) without slider test; creates false holds
  in rotation.
- **IHD** before measuring release transient with SRB alone.
- **VEG + HFQC + ORB** all as hard gates without ablation order (over-fitting
  thresholds to one room).
