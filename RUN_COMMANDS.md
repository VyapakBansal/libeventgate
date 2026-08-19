# Run commands (WSL)

All paths from repo root. Rebuild after pulling pipeline changes, then run
in order. On a **new bag**, change `session_1136` / `session_NEW` names only;
figure filenames stay the same.

## 0. Environment

```bash
cd /mnt/c/Projects/intelligent-mapping-navigation-lab/libeventgate

export TENSORRT_ROOT="${HOME}/sdks/TensorRT-10.16.1.11"
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH:-}"

# Figure scripts (once per machine; WSL blocks system pip — use venv)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# prompt should show (.venv). Then `python` / `python3` is the venv.
```

## 1. Build (once after code changes)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..
```

## 2. Three reconstructions (this bag)

**Ungated** (Section VII-A/B):

```bash
./build/eventgate --events eventgate_usb/events.h5 \
  --imu eventgate_usb/imu.csv --engine engines/firenet.engine \
  --out out/session_1136 --blackout-tail-s 2 --vram-every 50
```

**Freeze-h ablation** (Table II only; not the product):

```bash
./build/eventgate --events eventgate_usb/events.h5 \
  --imu eventgate_usb/imu.csv --engine engines/firenet.engine \
  --ablate-freeze-h --gyro-thresh 0.08 \
  --out out/session_1136_gated --blackout-tail-s 2 --vram-every 50
```

**Released latch + TEDG + CMDG + SRB** (orange series in figures):

```bash
./build/eventgate --events eventgate_usb/events.h5 \
  --imu eventgate_usb/imu.csv --engine engines/firenet.engine \
  --gate --gyro-thresh 0.08 \
  --hold-lookback-s 1 --hold-min-static-s 1 --hold-release-s 0.25 \
  --out out/session_1136_enhanced --blackout-tail-s 2 --vram-every 50
```

Extensions are **on by default** with `--gate`. Disable individually:

```bash
# examples
--no-tedg --no-cmdg --no-srb
--tedg-alpha 0.5 --com-drift-px 3 --release-blend-n 10
```

## 3. Figures + metrics

```bash
source .venv/bin/activate   # skip if the prompt already shows (.venv)

python3 scripts/plot_imu_hold_paper.py \
  --manifest eventgate_usb/MANIFEST.json \
  --imu eventgate_usb/imu.csv \
  --keypoints out/session_1136/keypoints.csv \
  --keypoints-gated out/session_1136_gated/keypoints.csv \
  --keypoints-lookback out/session_1136_enhanced/keypoints.csv \
  --video-ungated out/session_1136/recon.mp4 \
  --video-gated out/session_1136_gated/recon.mp4 \
  --video-lookback out/session_1136_enhanced/recon.mp4 \
  --out figures/imu_gated_evs_hold \
  --thresh 0.08 --n-stops 4 --min-stop-s 1

python3 scripts/draw_pipeline_fig.py

python3 scripts/stability_metrics.py \
  --video-ungated out/session_1136/recon.mp4 \
  --video-gated out/session_1136_gated/recon.mp4 \
  --video-lookback out/session_1136_enhanced/recon.mp4 \
  --out figures/imu_gated_evs_hold

cp figures/imu_gated_evs_hold/*.{png,pdf,json} paper/figures/
```

`stability_metrics.py` takes ~2 min (seeks three ~100 fps mp4s). Fill Table IV
from `figures/imu_gated_evs_hold/stability_metrics.json`.

## 4. New 240 s / four-stop bag

Replace `eventgate_usb/{events.h5, imu.csv, MANIFEST.json}`, retune
`--gyro-thresh` to the new median `||omega||`, then rerun sections 2–3 with
e.g. `out/session_NEW`, `out/session_NEW_gated`, `out/session_NEW_enhanced`.
Same script flags; only paths change.

## What the three extensions do

| Flag | Name | Role |
|------|------|------|
| (default on) | **TEDG** | Snapshot latch candidate when `n_k` decays before gyro static |
| (default on) | **CMDG** | Veto STATIC if event center-of-mass drifts > 3 px/window |
| (default on) | **SRB** | Blend held frame into first 10 live frames on release |

Math: `METHOD.md` Section "Event-native extensions" and paper Section IV-F
(`sec:extensions`).
