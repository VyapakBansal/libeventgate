#pragma once

#include "config.hpp"
#include "types.hpp"

#include <string>

namespace eventgate {

// HDF5 events -> CUDA voxels -> optional TensorRT FireNet -> ORB counts + video.
// Without an engine, voxels are summed to a grayscale proxy so the rest of the
// pipeline (HOLD, IMU gate, writers) can still run.

struct PipelineResult {
  int windows = 0;
  int static_windows = 0;
  std::string keypoint_csv;
  std::string video_path;
  float peak_vram_mib = 0.f;
};

PipelineResult run_pipeline(const PipelineConfig& cfg);

} // namespace eventgate
