# libeventgate

C++ / CUDA / TensorRT pipeline that reconstructs video from a **pure event camera** (no DAVIS grayscale) with FireNet, then **holds** that reconstruction when a MEMS gyroscope says the sensor is still.

This is not IMU-only navigation. The IMU is a still/moving bit, used the same way zero-velocity detection is used in a shoe-mounted INS: shut the state update when the platform is stationary.

Paper (Overleaf): [paper/main.tex](paper/main.tex). Markdown draft: [IMU_Gated_EVS_Static_Hold.md](IMU_Gated_EVS_Static_Hold.md).

If you use this code, cite the repository ([CITATION.cff](CITATION.cff)) and the paper.

```
V. Bansal, "libeventgate," GitHub, 2026. https://github.com/VyapakBansal/libeventgate
```

**Status.** This is research code for the paper, in the same class as [rpg_e2vid](https://github.com/uzh-rpg/rpg_e2vid): a working pipeline you can fork and run, not an SDK. There is no CI, no unit tests, no versioned C API, and no packaged FireNet weights. OpenEB / Metavision is what "industry event-camera software" looks like. Do not drop this into a vehicle stack as-is.

## What it does

1. Read events from HDF5 (`x, y, t_us, p`) and an IMU sidecar CSV.
2. Pack 10 ms, 5-bin voxel grids on the GPU.
3. Run FireNet in TensorRT (hidden state `h_in` / `h_out` must be exposed).
4. Two holds:
   - **Empty-window HOLD:** no events in the window → skip inference, re-emit the last frame.
   - **ARE gate + fade-horizon latch:** \(\|\omega\| < \delta\) → skip inference and latch a frame from the **fade horizon** (default 1 s). With `--gate`, **TEDG** (early snapshot on event decay), **CMDG** (COM drift veto), and **SRB** (soft release blend) are on by default.

`--events-left` and `--events-right` share one IMU so a stereo pair can freeze together. Sequential passes, one engine, so a 6 GB GPU is enough. Stereo numbers are not in the paper; the flags are.

## Why the hold lookback is 1 second

Not because “1 s looked good on a video.” It is the FireNet fade timescale.

Scheerlinck *et al.* (WACV 2020) and the ungated run in this repo both show reconstructions dying in about **1–2 s** of weak input. The default `--hold-lookback-s 1` is the **lower end of that window**:

| Horizon | What goes wrong |
|---------|-----------------|
| ~0.2 s | Still in the deceleration tail. Gyro is already near \(\delta\), events are already dying, FireNet is already gray. That is the frame people freeze by mistake. |
| **1 s** | Long enough to skip that tail. Short enough that the camera heading is still the stop pose. |
| ~2 s | Can latch a canvas from a different heading (the camera was still turning). |

Inside that 1 s buffer the code does **not** blindly take \(t-1\). It latches the **most recent** window whose event count is at least half the peak in the buffer: the last still-alive reconstruction, not the busiest (maybe smeared) one and not the last starved one.

`--hold-release-s 0.25` keeps the latch through 10 ms IMU flicker so one noisy gyro sample cannot run FireNet on empty voxels and wash the picture out.

`--hold-min-static-s 1` refuses to freeze until STATIC has lasted a full second. On this bag the median gyro-quiet bout is about 80 ms (24 Hz chatter). Those never latch. **Do not set this to 5 s:** FireNet is already gray after 1-2 s, so a 5 s wait freezes a dead frame. The lookback snapshot is taken at STATIC *onset*, then applied only if the stop lasts 1 s.

If a reviewer asks “why not cross-validate 0.5 / 1 / 2 s”: those three values are the short / fade / pose-mismatch set above. 1 s is the default because it matches the measured fade, not because it was fit to ORB.

## Not in this repository

| Thing | Why |
|-------|-----|
| `events.h5` / `events.raw` (~6 GB) | Too large for Git. Session `20260807_1136` is local only. |
| FireNet `.pth.tar` / TensorRT `.engine` | Third-party weights; get the checkpoint from the FireNet authors, then `scripts/fetch_firenet.sh` and `build_engine`. |
| Radar / DIRG / SLAM ATE | Out of scope. |

`eventgate_usb/imu.csv` (format + this session’s IMU) and `MANIFEST.json` (timing) are included so the paper plots can be redrawn.

## Build (WSL2 / Linux, CUDA + TensorRT)

```bash
export TENSORRT_ROOT="${HOME}/sdks/TensorRT-10.16.1.11"   # your install
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Dependencies: CMake 3.24+, CUDA 12.x, HDF5, OpenCV (core, imgproc, features2d, videoio), TensorRT 10.x for FireNet. Without TensorRT the voxel/smoke path still builds.

Default `CMAKE_CUDA_ARCHITECTURES=89` (Ada). Override for another GPU:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
```

## Run

Ungated reconstruction:

```bash
./build/eventgate --events eventgate_usb/events.h5 \
  --imu eventgate_usb/imu.csv --engine engines/firenet.engine \
  --out out/session_1136 --blackout-tail-s 2
```

ARE gate with fade-horizon latch + TEDG/CMDG/SRB (released default):

```bash
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH:-}"
./build/eventgate --events eventgate_usb/events.h5 \
  --imu eventgate_usb/imu.csv --engine engines/firenet.engine \
  --gate --gyro-thresh 0.08 --hold-lookback-s 1 --hold-min-static-s 1 --hold-release-s 0.25 \
  --out out/session_1136_enhanced --blackout-tail-s 2
```

Full step-by-step (build, three runs, figures): [RUN_COMMANDS.md](RUN_COMMANDS.md).

Figure scripts: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. After that, `python3` is the venv.

Freeze-h ablation (Table II only):

```bash
./build/eventgate --events eventgate_usb/events.h5 \
  --imu eventgate_usb/imu.csv --engine engines/firenet.engine \
  --ablate-freeze-h --gyro-thresh 0.08 \
  --out out/session_1136_gated --blackout-tail-s 2
```

Set `--gyro-thresh` from **your** IMU (this bag: median \(\|\omega\|\approx 0.087\) rad/s, \(\delta=0.08\)). Do not copy 0.08 onto another sensor.

HDF5 layout: datasets `/events/x`, `/events/y`, `/events/t_us`, `/events/p` (`uint16`, `uint16`, `int64`, `int8`). IMU CSV: `timestamp_us,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z`.

## License

MIT for this repository. FireNet / rpg_e2vid weights and code are **not** covered; follow their license when you download them.

## Acknowledgments

Intelligent Navigation and Mapping Laboratory (INML), Schulich School of Engineering, University of Calgary. Supervisor: Dr. Hongzhou Yang.
