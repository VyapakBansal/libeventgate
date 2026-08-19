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

  float gyro_static_thresh = 0.f;
  bool  gyro_thresh_set    = false; // --gate refuses to run until this is set

  std::string engine_path;
  std::string onnx_path;
  size_t trt_workspace_bytes = 512ull * 1024ull * 1024ull; // 512 MB; 6 GB cards OOM above this
  bool prefer_int8 = true;
  bool prefer_fp16 = true;

  std::string events_h5;
  std::string events_h5_right; // optional; same IMU/gate, sequential so one engine fits in VRAM
  std::string imu_csv;
  std::string out_dir = "out";
  std::string keypoint_detector = "ORB"; // ORB | FAST
  int  max_keypoints = 2000;
  bool write_video = true;
  bool write_keypoint_csv = true;
  bool enable_imu_gate = false;

  // Search window for the HOLD latch. FireNet fades in ~1-2 s of weak input;
  // 1 s skips the deceleration tail without grabbing a frame from a different heading.
  int64_t hold_lookback_us = 1'000'000;

  // Remain held until this much consecutive MOVING time. A single 10 ms gyro
  // spike would otherwise run FireNet on empty voxels and wash out the latch.
  int64_t hold_release_us = 250'000;

  // Do not latch until STATIC has lasted this long. Median gyro-quiet bout on
  // the 24 Hz sidecar is ~80 ms (chatter). 1 s matches the FireNet fade scale:
  // shorter stops have not yet starved the reconstructor. 5 s is too long;
  // the network is already gray by 1-2 s, so a 5 s wait freezes a dead canvas.
  int64_t hold_min_static_us = 1'000'000;

  // Extra empty windows after the last event so HOLD can run past end-of-stream.
  int64_t blackout_tail_us = 2'000'000;

  int vram_log_every = 50; // 0 = start/end only
  float vram_warn_mib = 4500.f;
};

PipelineConfig parse_cli(int argc, char** argv);
void print_config(const PipelineConfig& c);

} // namespace eventgate
