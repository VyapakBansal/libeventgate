#pragma once

// Phase 0 types — SoA event layout for coalesced H2D + kernels.

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <string>

namespace eventgate {

inline constexpr int kDefaultWidth    = 640;
inline constexpr int kDefaultHeight   = 512;
inline constexpr int kDefaultBins     = 5;
inline constexpr int kDefaultWindowUs = 10'000; // 10 ms

// SoA event batch — preferred GPU/host layout (not AoS).
struct EventSoA {
  const uint16_t* x    = nullptr;
  const uint16_t* y    = nullptr;
  const int64_t*  t_us = nullptr;
  const int8_t*   p    = nullptr; // +1 / -1
  int32_t         n    = 0;
};

// Common stream metadata for HDF5 / MCAP offline loaders.
struct EventStreamMeta {
  int width  = kDefaultWidth;
  int height = kDefaultHeight;
  std::string sensor;
  int64_t n_events = 0;
  int64_t t_min_us = 0;
  int64_t t_max_us = 0;
};

struct Event {
  uint16_t x;
  uint16_t y;
  int64_t  t_us;
  int8_t   p; // +1 / -1
};

struct VoxelShape {
  int bins   = kDefaultBins;
  int height = kDefaultHeight;
  int width  = kDefaultWidth;

  [[nodiscard]] size_t elements() const noexcept {
    return static_cast<size_t>(bins) * static_cast<size_t>(height) * static_cast<size_t>(width);
  }
  [[nodiscard]] size_t bytes_f16() const noexcept { return elements() * 2u; }
  [[nodiscard]] size_t bytes_f32() const noexcept { return elements() * sizeof(float); }
};

struct ImuSample {
  int64_t t_us = 0;
  float gyro[3]  = {0, 0, 0};
  float accel[3] = {0, 0, 0};

  [[nodiscard]] float gyro_norm() const noexcept {
    return std::sqrt(gyro[0] * gyro[0] + gyro[1] * gyro[1] + gyro[2] * gyro[2]);
  }
};

enum class GateState : uint8_t {
  Moving = 0,
  Static = 1, // freeze ConvGRU write-back
};

struct FrameMetrics {
  int64_t t0_us = 0;
  int64_t t1_us = 0;
  int     keypoint_count = 0;
  GateState gate = GateState::Moving;
  float   gyro_norm = 0.f;
  float   inference_ms = 0.f;
  float   voxel_ms = 0.f;
};

} // namespace eventgate
