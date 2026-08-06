#pragma once

#include "config.hpp"
#include "types.hpp"

#include <string>
#include <vector>

namespace eventgate {

// End-to-end offline Phase 0 pipeline:
//   HDF5 events → CUDA voxel (5–10 ms) → (optional TRT FireNet) → ORB counts → video/CSV
//
// Without a TRT engine, runs voxel + empty-frame blackout path so Day 1–2 can land hardware-free.

struct PipelineResult {
  int windows = 0;
  int static_windows = 0;
  std::string keypoint_csv;
  std::string video_path;
  float peak_vram_mib = 0.f;
};

PipelineResult run_phase0(const PipelineConfig& cfg);

} // namespace eventgate
