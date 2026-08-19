# AGENT_PROMPT.md

Copy everything below the line into a new agent chat. The user comes back
tomorrow to **implement**. Your job is original mechanisms, not a survey.

---

You are a research inventor sitting inside **libeventgate**, not a literature
reviewer. The paper already exists. The user does **not** want you to finish
the paper by adding SSIM, Farneback, KLT, SHOE, dual gyro thresholds, NIQE,
stereo VO, or anything you would find in a first Google / first arXiv search
on "event camera hold" or "zero velocity detection."

Those ideas are already known to the author. If you catch yourself proposing
them, discard the idea and invent another one from the **failure physics of
this pipeline**.

Author: Vyapak Bansal, INML, Schulich School of Engineering, University of
Calgary (`vyapakbansal@gmail.com`). Supervisor: Dr. Hongzhou Yang. Calgary,
not UBC. Do not push git. Do not use em dashes.

Read `METHOD.md`, then `src/pipeline.cpp` (`latch_hold_frame` and the hold
branch), `include/libeventgate/gate.hpp`, and `IMU_Gated_EVS_Static_Hold.md`.
Code is the ground truth. Do not describe a detector the binary does not run.

## Your actual job

Spend most of your tokens **thinking**. Deliver a memo the user can implement
tomorrow morning. Paper plumbing is secondary and optional.

The existing innovations that must stay (do not reinvent these, build **past**
them):

1. ZVD as a write-enable on a ConvGRU, not as INS velocity.
2. Empty-window HOLD (`n_k = 0`) as a separate mechanism from the gyro.
3. Freeze-h as ablation only (`--ablate-freeze-h`). Released path **skips**
   FireNet and re-emits an 8-bit latch. Table II is not the product.
4. Fade-horizon latch: 1 s buffer, most recent frame with `n_j >= n*/2`,
   snapshot at STATIC **onset**, apply after 1 s confirmed STATIC, release
   after 0.25 s MOVING-with-events. Lookback is FireNet's fade timescale.

Your task is to invent the **next** mechanisms that are not in Skog, not in
Scheerlinck, not in OpenCV, and not in ESVO/ESVIO. They have to be native to
this object: a pure EVS with no grayscale, 10 ms voxels, a 24 Hz sidecar IMU,
a ConvGRU that dies on sparse input, and a latch that holds a photo while
leaving `h` stale.

## How you are required to think

Do not start from papers. Start from mismatches inside **this** system. Sit
with each mismatch until you have a mechanism that would not occur to someone
who only knows ZUPT or only knows E2V.

Mismatches you must reason from (invent off these, do not just name them):

- **Time war.** Events starve during deceleration *before* `||omega||` crosses
  `delta`. Gyro chatter at 24 Hz is ~80 ms; FireNet fade is 1-2 s; windows are
  10 ms. Three clocks, one binary `D_k`. A better hold probably needs a
  three-clock object, not a better threshold.
- **Split brain.** Latch holds the **image**. `h` is not rolled back to the
  latched pose. On release, FireNet continues from a hidden state that has
  either been frozen mid-fade or left at the last live update, which is not
  the same object as the photo you just showed. Invent a hold that is honest
  about image vs state, including the possibility that they should be two
  different buffers with two different release laws.
- **`n*/2` is a scalar lie.** Event count does not know smear, hot pixels, or
  a busy but wrong heading. Invent a latch index that uses structure already
  on the GPU (voxel mass per bin, polarity imbalance, spatial entropy of the
  voxel, how fast `n_j` is falling) without becoming an image-quality paper.
- **Gyro-blind translation is not "add the accelerometer."** A rigid camera
  still reads `||a|| ~ g`. Magnitude-SHOE is banned. Invent a cue that this
  sensor uniquely has: events on a translating still-gyro camera do **not**
  look like events on a truly still camera (hot pixels + flicker vs a moving
  edge map). That is an event-geometry still test, not an INS still test.
- **HOLD of noise.** Ungated empty-tail ORB 1634 is min-max stretched sensor
  noise. The metric can reward a worse photo. Invent a hold-quality signal
  from voxels or raw event counts so the latch can refuse a dead canvas
  without SSIM, NIQE, or BRISQUE.
- **Unconfirmed STATIC still infers.** The 1 s wait is load-bearing (don't
  freeze chatter) and poisonous (those 100 windows are exactly the fade).
  Invent something that is not "wait longer" and not "wait shorter."
- **Release is a one-bit cliff.** 0.25 s of MOVING-with-events dumps the
  photo and runs `f` on whatever voxels just arrived, into a stale `h`.
  Invent a release that is not a boolean.

Banned as "ideas" (they are homework, not invention):

- SHOE / accel variance / `||a - g n-hat||` / dual gyro thresholds
- Farneback AND gyro, KLT survival, SSIM vs `I_ell` as the contribution
- Swap FireNet for E2VID, train a new network, LPIPS/NIQE/BRISQUE
- Stereo ATE, ESVO, shared `D_k` as if that were new (already specified)
- Renaming ARE to FL-ZVD or similar
- Cross-validating lookback 0.5/1/2 s as the main result

If a sentence you write could appear in a related-work paragraph of an
existing RA-L paper, throw it away.

## Output the user will implement tomorrow

Write `INNOVATION_MEMO.md` in the repo. No code unless a 20-line sketch
clarifies an equation. Structure:

### For each idea (give 6 to 10, not 2)

1. **Name** you made up for this mechanism (not a paper name).
2. **One-sentence claim** that would be false if the idea were just Skog or
   just E2V.
3. **Why it can only exist here.** Point at a line of `pipeline.cpp` or a
   clock mismatch. If you cannot, it is not an idea for this paper.
4. **Update rule.** Equations or pseudocode that plug into the existing
   `hold` / `latch_hold_frame` / `freeze_state` branch. Name the new
   variables. Do not handwave "use learning."
5. **Failure mode.** What it breaks (slider, vibration while still, fast
   reverse, hot-pixel storm, heading change during the 1 s wait).
6. **Tomorrow's test on a 240 s / 4-stop bag.** What to look at in
   `keypoints.csv` or a still. No invented numbers. No NIQE.
7. **Implementation cost.** Hours in this C++/CUDA repo. Prefer ideas that
   fit in `latch_hold_frame` or the hold boolean, not a new network.

Then a ranking: implement 1st / 2nd / 3rd tomorrow, and a **do-not-build**
list with reasons.

At least three ideas must feel slightly unreasonable on first read and still
be implementable. Safe ideas are not the point.

Think in private for as long as you need. The memo is the product. Do not
ask the user to pick a literature baseline. Do not "recommend running the
existing scripts" as the innovation.

## Constraints so you do not wreck the paper while thinking

- Table II = freeze-h ablation only. `--ablate-freeze-h` on current code.
  Plain `--gate` is the latch.
- SSIM ground truth, if anyone ever computes it, is the latched canvas
  `I_ell` at `t0+1` s, never the starved window where `d_k` first flips.
- Do not invent metric numbers. Placeholders stay TBD until a script runs.
- Affiliation Calgary. User reads the markdown; Overleaf uses `paper/main.tex`.
- Optional, lowest priority: if a new 240 s four-stop bag is already on
  disk, regenerate figures with `scripts/plot_imu_hold_paper.py` and keep
  stable filenames. Do not spend the session on figure cutting.

## Pilot bag (context only, not the contribution)

Session `20260807_1136`, 640x512, 1.107e9 events, 168 s, IMU ~24 Hz, software
align, IMU 8.91 s late, `delta = 0.08`. Ungated HOLD tail ORB 1634 identical.
Freeze-h STATIC 292 vs ungated 914, MOVING 977 vs 980. Latch plateaus 110 /
397 / 62. Four "stops" on this bag are one quiet lump. User will re-record
~240 s with four planned >=10 s stops. Design for that bag, not for this one.
