# Product Requirements Document (PRD)

**Project Name:** libeventgate — Doppler-Inertial Gated Neural Reconstruction Engine
**Document Status:** Research & Development Phase
**Target:** High-speed autonomous SLAM systems, robotics research, demo for Dr. Hongzhou Yang (INML)
**Timeline:** ~1 week pitch-prep (no radar needed) + 4-6 week build after approval, ~10 hrs/day
**Primary Deliverable:** Phase 0 — a pitch demo to get the radar purchase approved. Phase 1 — working demo (quad-view dashboard + keypoint survival comparison) once radar arrives.

---

## 1. Executive Summary & Vision

libeventgate is a C++ / CUDA library that stabilizes event-to-video reconstruction during low- and zero-motion phases, for any downstream task that consumes the reconstructed frames — SLAM keypoint tracking, but also object detection, place recognition, loop closure, and visual servoing, all of which suffer the same starvation problem when the reconstruction decays. It uses a TensorRT-optimized FireNet (Recurrent Convolutional Neural Network) to reconstruct dense intensity frames from asynchronous event data in real time, and demonstrates the gating mechanism against an ORB-SLAM3 backend as the primary validation case.

**Core differentiator:** Doppler-Inertial Recurrent State Gating (DIRG) — a general mechanism for gating an event-to-video network's recurrent hidden state ($h_t$) using an external absolute-velocity signal, independent of what that signal comes from. This build fuses FMCW radar Doppler velocity with IMU angular rate as the reference implementation, but the gate is designed against a pluggable velocity-source interface — wheel odometry, RTK-GPS, or sparse optical flow from a secondary camera would all satisfy the same interface. Radar is the chosen source here because it's absolute (no dead-reckoning drift) and because many platforms already carry one for obstacle avoidance or ego-velocity estimation, making the gating effectively free hardware-wise on those platforms. The gate also isn't strictly binary: FireNet's hidden-state update rate scales inversely with velocity magnitude, so it functions as a continuous regularizer across slow-motion regimes (panning, hovering, crawling), not just a hard freeze at a full stop. Either way, it prevents the state decay that normally causes reconstructed frames — and everything downstream that depends on them — to collapse to noise within 1-2 seconds of standstill.

---

## 2. Novelty & Related Work (for supervisor review)

This section exists specifically to answer "does this already exist" before any hardware is ordered.

**The problem is real and documented, not a novel claim on its own.** The zero-motion blackout in FireNet/E2VID-style event-to-video reconstruction is an acknowledged limitation in the original FireNet paper (Scheerlinck et al., WACV 2020) and follow-ups — recurrent hidden states decay without new events, and reconstructions fade to gray/noise within 1-2 seconds of standstill.

**Closest prior art, and how this differs:**

| Work                                                                                                                                    | What it does                                                                                   | Why it isn't this                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Safa et al. 2023, "Fusing Event-based Camera and Radar for SLAM Using Spiking Neural Networks with Continual STDP Learning" (ICRA 2023) | Fuses DVS + FMCW radar for drone SLAM via a spiking neural network that learns features online | Replaces the whole vision pipeline with an SNN; doesn't gate a CNN's recurrent state or plug into conventional SLAM backends              |
| mmE-Loc, "Ultra-High-Frequency Harmony" (2025)                                                                                          | mmWave radar + event camera fusion                                                             | Task is drone-landing ground localization, not general E2V reconstruction                                                                 |
| NERVE (2026)                                                                                                                            | DVS-radar fusion dataset/benchmark                                                             | Explicitly states DVS-radar fusion is still an "open research direction lacking established baselines"                                    |
| TwistEstimator / UNRIO                                                                                                                  | Radar-event and radar-inertial ego-velocity estimation                                         | Estimate velocity states directly; don't touch a neural network's internal hidden state                                                   |
| ZUPT literature (INS/pedestrian nav, decades old)                                                                                       | Freezes velocity/position error states when an IMU detects standstill                          | Applied to filter states in inertial navigation, never to a ConvGRU/ConvLSTM hidden state inside an event-to-video reconstruction network |

**The gap:** no existing work fuses absolute Doppler velocity + IMU angular rate to gate the write-back of a recurrent hidden state inside an event-to-video reconstruction network, specifically to preserve ORB/FAST keypoints for a downstream conventional (non-learned) SLAM backend. Multiple targeted searches (E2V + radar gating, hidden-state freezing + zero-motion, ZUPT + RNN state) turned up nothing doing this combination.

**Framing for the pitch to Dr. Yang:** don't claim "radar+event fusion is new" (it isn't — cite Safa et al. and mmE-Loc as related work). Claim that _gating a reconstruction network's hidden state with an absolute-velocity signal to preserve downstream keypoint survival_ is the new mechanism, and that it's a lightweight augmentation to an existing pipeline rather than a from-scratch learned system — which is also the practical argument for why it's buildable in 4-6 weeks.

**On impact — the honest version, worth raising proactively rather than waiting for him to ask:** the zero-motion blackout itself is a narrow problem — most systems already tolerate short stops via IMU dead-reckoning, so this matters most for _long_ stops (drift accumulates) and for platforms where swapping in a conventional camera isn't a viable alternative (harsh environments — fog, dust, extreme low light — where the event camera was chosen because regular cameras also struggle there). Two design choices in this PRD exist specifically to broaden that narrow case rather than overclaim it:

1. **Radar is one implementation of a general velocity-gated interface (P1),** not a hard dependency — any absolute-velocity source works, which widens the applicable-platform set well beyond "robots with radar."
2. **The fix generalizes past SLAM (P1a)** to any task consuming reconstructed frames — SLAM keypoint survival is this build's measurable proxy, not the ceiling of the contribution.
   The strongest version of the impact story is platforms that already carry radar for another purpose (obstacle avoidance, ego-velocity) getting this gating essentially for free.

**Open risk to flag to him directly:** this is a novelty check via search, not a formal literature review or patent search. Before committing further (especially before a paper submission), do a proper pass through IEEE Xplore / Google Scholar citation graphs of the five papers above, since search engines can miss workshop papers, theses, and non-indexed preprints.

---

## 3. Core Problem Statement

- **Zero-Motion Blackout:** Event cameras (e.g. IMX637) rely on dynamic illumination changes. When camera motion stops, event generation ceases.
- **Recurrent State Decay:** Standard E2V networks (FireNet, E2VID) expect continuous event streams. During a stop, ConvGRU/ConvLSTM hidden states decay numerically; reconstructed video fades to gray or fills with shot noise within ~1-2 seconds.
- **SLAM Divergence (primary validation case):** Visual SLAM extracts keypoints (ORB/FAST) from these frames. When the reconstruction decays, the SLAM backend loses visual landmarks, causing trajectory drift or localization failure. The same starvation affects any other downstream consumer of the reconstructed frames — detection, place recognition, loop closure — but SLAM keypoint survival is this project's measurable target.

---

## 4. Hardware Requirements (Bill of Materials)

| Subsystem              | Hardware                                                                                                       | Status                                                                        | Function                                                                                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Event Camera           | LUCID Triton2 EVS (Prophesee IMX637)                                                                           | In hand (from stereo calibration rig)                                         | Generates HDR (120dB+) asynchronous events via 2.5GigE                                                                                                                        |
| FMCW Radar             | TI IWR6843AOPEVM (60 GHz)                                                                                      | **Not ordered — pending pitch approval (Phase 0), ETA 2-3 weeks after order** | Absolute, drift-free Doppler velocity to trigger the static gate                                                                                                              |
| IMU                    | (per your existing rig)                                                                                        | In hand                                                                       | High-frequency angular rate ($\omega$) and linear acceleration                                                                                                                |
| Compute Node           | Windows 11 laptop: Intel Core 7 240H (10C/16T), 16GB DDR5-5600, RTX 4050 Laptop (6GB VRAM), 1TB NVMe, via WSL2 | Confirmed                                                                     | TensorRT inference, CUDA voxel kernels, OpenGL interop                                                                                                                        |
| Secondary/Linux laptop | i5 11th gen, 16GB RAM, 1.5TB storage, integrated graphics only                                                 | Confirmed — no dGPU, not for TensorRT/CUDA work                               | **Primary capture machine**: native event/IMU/radar recording (~15GB/session), transferred to Windows for processing. Also camera SDK bring-up, docs, radar UART parsing code |

**Two-machine workflow:** capture and compute are split across the two laptops, not co-located. Record raw event streams + IMU (and radar, once it arrives) natively on the Linux laptop — native Metavision SDK timing is more reliable than through WSL2's USB/network passthrough. Transfer each recording to the Windows laptop over WiFi via `rsync` (SSH key auth set up once, then one command per session — see workflow note below), and run reconstruction, gating, and the dashboard against the recording there. This also means every ablation comparison (baseline vs. IMU-only vs. full DIRG) runs against the _same recorded motion_, not three separate live captures — which is actually better methodology for Section 8's evaluation protocol than live-only testing would have been.

**Transfer:** `rsync -avz --progress` over SSH, key-based auth. ~15GB/session is a few minutes on LAN WiFi; resumable if it drops mid-transfer.

**VRAM constraint:** 6GB is tight for this pipeline running end-to-end (FireNet + voxel buffers + hidden state + TensorRT workspace + OpenGL/CUDA-interop buffers for the quad-view dashboard, all concurrent). Treat INT8 quantization (already listed under P0) as required, not optional, and budget time in Phase 0 to profile VRAM usage early rather than discovering an OOM in Week 3.

**Critical path note:** the radar purchase itself is gated on Dr. Yang's approval, which is gated on Phase 0 (Section 9) proving the IMU alone isn't enough. Order it the moment Phase 0 lands.

---

## 5. Software Stack & Dependencies

- **Camera SDK:** Prophesee Metavision SDK (v4.6 or v5.x) + LUCID Arena SDK C++ API
- **Machine Learning:** NVIDIA TensorRT 10.x C++ API (pin 10.8+ for CUDA 12.8 / Ada sm_89 full FP16+INT8), CUDA 12.x toolkit inside WSL2. Workspace hard-capped at **512 MB** for 6 GB VRAM; prefer FP16 until INT8 calibrator exists
- **Event interchange (Phase 0):** Prophesee RAW on capture laptop → convert to HDF5 for rsync/processing (decouples SDK versions). Full IMX637 resolution at capture; crop/downsample only in the voxel path if VRAM forces it
- **IMU interchange (Phase 0):** sidecar CSV next to the event file — columns `timestamp_us,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z`, same clock domain as the event stream
- **FireNet weights:** official Scheerlinck et al. WACV 2020 — `cedric-scheerlinck/rpg_e2vid` branch `cedric/firenet`, checkpoint `firenet_1000.pth.tar` (Google Drive id `1nBCeIF_Us-rGhCjdU5q1Ch-yrFckjZPa`; PyTorch → ONNX with externalized ConvGRU `h_in_*`/`h_out_*` → TensorRT; plain FireNet, not FireNet+)
- **Sensor Fusion:** Custom lock-free C++ ring buffers mapping UART/serial radar data
- **Visualization:** OpenGL, GLFW (`cuda_gl_interop.h` for zero-copy rendering)
- **SLAM Backend Integration:** ROS 2 (Humble) custom messages or direct shared-memory pointers to ORB-SLAM3/GTSAM

---

## 6. Functional Requirements

### P0: Real-Time Event-to-Video Engine (Baseline)

- Continuous voxelization: convert asynchronous `EventCD` arrays into 3D voxel grid tensors inside a custom CUDA kernel every 5-10 ms
- TensorRT inference: run a PyTorch FireNet model converted to ONNX/TensorRT `.engine` (FP16 or INT8)
- Zero-copy memory: CPU-to-GPU transfers via pinned memory (`cudaHostAlloc`)

### P1: Doppler-Inertial Recurrent State Gating (DIRG) — Primary Research Novelty

- **Velocity-source interface:** the gate consumes an abstract absolute-velocity signal ($v$) plus IMU angular rate ($\omega$), not radar specifically. FMCW radar is the concrete implementation for this build (`RadarVelocitySource`), but the interface is designed so wheel odometry, RTK-GPS, or optical-flow-derived velocity could satisfy it without touching the gating logic — this is what makes DIRG a general mechanism rather than a radar-specific patch.
- **Hard gate (baseline mode):** if $v < \epsilon$ AND IMU gyro $< \delta$, enter `STATIC_MODE`
- **Soft gate (extended mode):** rather than a binary freeze, scale the ConvGRU hidden-state update rate inversely with $|v|$ — full update at speed, fully frozen at $v = 0$, smoothly interpolated between. This extends the mechanism from "fixes full stops" to "regularizes any low-motion regime" (slow pans, hovering, crawling), which is a meaningfully larger operating envelope than zero-motion alone. Build the hard gate first (Phase 1); treat soft gating as a stretch goal once the hard gate is validated.
- CUDA state lock: during `STATIC_MODE` (hard gate) or at reduced update rate (soft gate), the TensorRT context bypasses or throttles the write-back phase of the ConvGRU hidden state ($h_t$); the held state ($h_{t-1}$) persists in GPU VRAM
- **Fallback/ablation build order:** implement an IMU-only gate first (radar not yet available) as both a development stopgap and the "why not just use IMU" ablation the paper/demo needs anyway

### P1a: Downstream Consumers Beyond SLAM (framing, not build scope)

The gated reconstruction is validated against ORB-SLAM3 keypoint survival (Section 8) as the primary demonstration, but the underlying problem — and therefore the fix — isn't SLAM-specific. Object detection, place recognition, loop closure, and visual servoing all consume reconstructed frames and starve the same way during decay. Worth stating explicitly in the pitch and any eventual paper: the contribution is a general reconstruction-stabilization mechanism, with SLAM keypoint survival as the measurable proxy for this build's scope, not the ceiling of the idea.

### P2: Zero-Latency Dual-Canvas Rendering

- Quad-view OpenGL dashboard:
  - Top-left: raw event polarity stream (red/blue)
  - Bottom-left: standard FireNet reconstruction (baseline, allowed to decay)
  - Top-right: gated FireNet reconstruction (proposed, locked during zero-motion)
  - Bottom-right: sensor telemetry (Doppler velocity curve, IMU state, gating trigger LED)
- CUDA-GL interop: render directly from GPU memory via `cudaGraphicsGLRegisterImage`

### P3: SLAM Covariance Output

- When `STATIC_MODE` is active, publish an uncertainty mask / high-covariance flag to the ORB-SLAM3 backend, instructing the optimizer to trust radar/IMU over visual keypoints

---

## 7. Non-Functional Requirements & Performance Targets

| Metric                 | Target     | Rationale                                                                                                 |
| ---------------------- | ---------- | --------------------------------------------------------------------------------------------------------- |
| Pipeline Latency       | < 2.5 ms   | Voxel prep (~0.2ms) + inference (~1.2ms) + render (~0.3ms); needed for closed-loop robotic control        |
| Framerate Throughput   | ≥ 100 FPS  | SLAM tracking needs dense temporal information                                                            |
| SSIM During Standstill | > 0.90     | Baseline SSIM drops below 0.20 after 2s; gated model must hold structural similarity for 60+ second stops |
| Hardware Resilience    | Drop-proof | Deployable on rovers/UGVs without losing radar-camera clock sync                                          |

---

## 8. Research Evaluation Protocol

Log metrics autonomously during test drives (e.g. rover navigating a room, stopping for 30s):

1. **Keypoint Survival Rate:** graph active ORB keypoints over time. Expect baseline → 0 during stops; gated → stays > 400.
2. **Absolute Trajectory Error (ATE):** compare SLAM trajectory against ground truth (Vicon/OptiTrack if available — flag if not, and what substitute you'll use).
3. **Ablation Study:** prove IMU-only gating fails (accelerometer bias drift) to justify the FMCW radar. This ablation is buildable in Week 1-2, before radar arrives — it's your IMU-only baseline.

---

## 9. Execution Timeline

Split into two phases: **Phase 0** gets you approval to buy the radar. **Phase 1** is the full build once it arrives. Nothing in Phase 0 requires the radar or a resolved ATE/sync decision — those are correctly deferred to Phase 1.

### Phase 0 — Pitch Demo (Days 1-6, no radar, no fixed deadline but move fast)

**Goal:** don't ask Dr. Yang to trust that IMU-only gating is insufficient — show him. The pitch is: baseline blackout is real → IMU-only gating fixes the easy case → IMU-only gating has a specific, demonstrable failure mode → that failure mode is exactly what absolute radar velocity closes → here's the literature gap (Section 2) → approve the ~2-3 week radar order.

- **Day 1:** Set up dev environment (CUDA, TensorRT, Metavision SDK, LUCID Arena SDK). Confirm compute node access.
- **Days 2-3:** Stream the Triton2 EVS over 2.5GigE, build the CUDA voxelization kernel, convert FireNet to a TensorRT engine, get baseline reconstruction running. **Record the zero-motion blackout on your own hardware** — this is your "before" clip and the first thing Dr. Yang sees.
- **Days 4-5:** Wire up the IMU. Build the IMU-only static gate (gyro threshold $\delta$) and the CUDA state-lock (bypass ConvGRU write-back). Then deliberately test it against the case that breaks it — pick one:
  - **Slow near-constant-velocity translation:** the IMU's gyro sees ~zero rotation, so the gate falsely triggers `STATIC_MODE` while the camera is actually moving — the reconstruction freezes on a stale frame while the scene changes underneath it, which SLAM will treat as a false-static keyframe.
  - **Long static hold (60s+):** gyro/accelerometer bias drift causes the gate to flicker (unfreeze spuriously), letting the hidden state decay anyway during what should be a clean freeze.

  Either one gives you a concrete, filmable failure — pick whichever is easier to stage with what you have. This is the ablation argument, just moved earlier and used as leverage instead of a paper section.

  **Which one to pick:**

  |                                      | Slow constant-velocity translation                                                                                                                                  | Long static hold (60s+)                                                                                                    |
  | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
  | What it needs                        | A way to move the rig at a slow, steady speed in a straight line (rolling cart, slider, or just a very controlled hand-carry)                                       | Nothing but time — set it down and wait                                                                                    |
  | Rig complexity                       | Higher — the motion has to be genuinely low-rotation, or the gyro will correctly reject it and you won't get the failure                                            | Lower — no motion control needed at all                                                                                    |
  | How obvious the failure is on camera | Very obvious — the scene visibly changes while the reconstruction stays frozen, easy to narrate in 10 seconds                                                       | More subtle — you're watching a slow flicker/decay over a minute, needs a keypoint-count graph to land the point clearly   |
  | Time to stage                        | 1-2 tries once you have a way to move smoothly                                                                                                                      | 1 take, but you're waiting out the full 60s+ each time you re-shoot                                                        |
  | Recommendation                       | **Pick this if you have any way to get smooth linear motion** (even a desk chair rolled at constant speed works) — it's the more visually convincing 10-second clip | Fall back to this only if you can't get clean linear motion — it's real, but needs the graph to sell it, not just the clip |

- **Day 6:** Package it: a 2-3 minute demo (live or recorded) — baseline blackout → IMU-gate succeeding on an easy stop → IMU-gate failing on the case above, keypoint count crashing anyway → the Section 2 novelty table → the broadened-impact framing (pluggable velocity source, beyond-SLAM applicability) → the ask (approve the IWR6843AOPEVM order, 2-3 week lead time). Dry-run the pitch once before the real one.

### Phase 1 — Full Build (starts on approval; 4-6 weeks, ~10 hrs/day)

**Week 1 (post-approval) — Order Radar, Dashboard, SLAM Hookup**

- **Day 1:** Order TI IWR6843AOPEVM the moment you get the green light.
- **Days 2-4:** Build the quad-view OpenGL dashboard (raw events, baseline recon, gated recon, telemetry panel) with CUDA-GL interop, reusing the Phase 0 pipeline.
- **Days 5-7:** Integrate ORB-SLAM3 (shared-memory or ROS2 topic). Get keypoint counts flowing into the telemetry panel. Re-run the Phase 0 IMU-only comparison here as your formal in-progress ablation.

### Week 2 — Waiting on Radar, Prep Work

- **Days 8-11:** Write the radar UART parsing thread against the TI datasheet / simulated serial data so it's ready to plug in the moment hardware lands.
- **Days 12-14:** Harden the IMU-only gate, tune keypoint-survival logging, start drafting the demo narrative structure for the eventual full pitch.

### Week 3 — Radar Arrives, DIRG Integration

- **Days 15-16:** Bring up the IWR6843AOPEVM, verify Doppler velocity readout over UART, calibrate velocity threshold $\epsilon$.
- **Days 17-18:** Combine radar + IMU into the full DIRG gate logic. Sync radar/camera/IMU timestamps.
- **Days 19-21:** Re-run keypoint survival and SSIM tests with full DIRG gating. Compare against IMU-only and ungated baselines — this is your three-way ablation table.

### Week 4 — SLAM Covariance Output, Tuning, ATE

- **Days 22-24:** Implement P3 (uncertainty mask / covariance flag to ORB-SLAM3 optimizer).
- **Days 25-26:** Performance tuning to hit the < 2.5ms / ≥100 FPS targets. Profile and fix bottlenecks.
- **Days 27-28:** Run ATE comparison against ground truth (confirm ground-truth method — Vicon/OptiTrack or a substitute — before this point, it'll block the whole step if not sorted).

### Week 5 — Demo Polish & Buffer

- **Days 29-31:** Buffer for whatever slipped in Weeks 1-4 (something will).
- **Days 32-33:** Polish the quad-view demo, record clean demo footage, generate the keypoint-survival and SSIM graphs for the ablation table.
- **Days 34-35:** Dry-run the pitch to Dr. Yang: problem statement → related work gap (Section 2) → live demo → ablation results.

### Week 6 (if using the 6-week option)

- Additional field tests on a rover/UGV, robustness testing (lighting, longer stops), and start a paper outline if the demo lands well.

---

## 10. Risks & Mitigations

| Risk                                                                            | Mitigation                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Radar shipping delays beyond 3 weeks                                            | Phase 0 already produces a complete IMU-only pipeline and demo as a fallback deliverable                                                                                                                                    |
| No motion-capture ground truth available                                        | Not needed for Phase 0. Decide a substitute (e.g. fiducial markers, fixed checkerboard path) by end of Phase 1 Week 2, not Week 4                                                                                           |
| TensorRT conversion issues with FireNet's ConvGRU layers                        | Budget Phase 0 Days 2-3 explicitly for this; ConvGRU custom layers sometimes need manual TensorRT plugin work                                                                                                               |
| Camera-radar-IMU clock sync drift                                               | Only camera+IMU sync matters for Phase 0 (simpler, software timestamp alignment is enough). Confirm the full 3-sensor hardware sync method (PTP, hardware trigger, or software alignment) before Phase 1 Week 3, not during |
| 6GB VRAM budget exceeded once dashboard + SLAM + full pipeline run concurrently | Profile VRAM in Phase 0 Days 2-3 alongside baseline reconstruction, not later. INT8 quantization is the primary lever; consider rendering the quad-view at reduced resolution if still tight                                |
| Related-work gap turns out incomplete on closer literature review               | Section 2 flags this explicitly — do a proper Scholar/IEEE pass before any paper submission, independent of the demo timeline                                                                                               |
