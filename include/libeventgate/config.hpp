#pragma once

#include "types.hpp"

#include <cstdint>
#include <string>

namespace eventgate {

struct PipelineConfig {
  int width  = kDefaultWidth;
  int height = kDefaultHeight;
  int bins   = kDefaultBins;
  int window_us = kDefaultWindowUs;

  // IMU hard gate; NaN means "unset — treat as always Moving until user sets it"
  float gyro_static_thresh = 0.f;
  bool  gyro_thresh_set    = false;

  // TensorRT
  std::string engine_path;
  std::string onnx_path;
  size_t trt_workspace_bytes = 512ull * 1024ull * 1024ull; // 512 MB hard cap
  bool prefer_int8 = true;
  bool prefer_fp16 = true;

  // I/O
  std::string events_h5;
  // Optional second EVS stream. Same IMU + ImuHardGate; sequential pass, shared timestamps.
  std::string events_h5_right;
  std::string imu_csv;
  std::string out_dir = "out";
  std::string keypoint_detector = "ORB"; // ORB | FAST
  int  max_keypoints = 2000;
  bool write_video = true;
  bool write_keypoint_csv = true;
  bool enable_imu_gate = false;

  // Hold lookback horizon (default 1 s). This is the FireNet fade timescale
  // (reconstructions die in about 1-2 s of weak input), not a visual tweak.
  // Shorter (~0.2 s) still sits in the deceleration tail. Longer (~2 s) can
  // freeze a canvas from a different heading. Inside the horizon the pipeline
  // latches the most recent window with at least half the peak event count.
  int64_t hold_lookback_us = 1'000'000;

  // Stay held until gyro is MOVING for this long. Stops 10 ms IMU flicker from
  // running FireNet on empty voxels and washing the held frame out.
  int64_t hold_release_us = 250'000;

  // After last event, extend timeline with empty windows (easy-stop blackout demo tail).
  // Default 2 s matches Day-1 smoke / pitch "before" clip packaging.
  int64_t blackout_tail_us = 2'000'000;

  // Log VRAM every N windows (0 = only start/end + on load).
  int vram_log_every = 50;

  // VRAM soft budget warn (MiB)
  float vram_warn_mib = 4500.f;
};

// Minimal CLI: --events [--events-right] --imu --engine --out --window-ms --gyro-thresh --gate --no-video
PipelineConfig parse_cli(int argc, char** argv);
void print_config(const PipelineConfig& c);

} // namespace eventgate
