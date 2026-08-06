#pragma once

#include "types.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace eventgate {

inline void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

// Device buffers for one voxel volume + staging events (double-buffered).
class VoxelizerCuda {
public:
  VoxelizerCuda(VoxelShape shape, int max_events_per_window, int stream_priority = 0);
  ~VoxelizerCuda();

  VoxelizerCuda(const VoxelizerCuda&) = delete;
  VoxelizerCuda& operator=(const VoxelizerCuda&) = delete;

  // Pinned host staging for events in the active window (AoS-free SoA).
  struct HostStaging {
    uint16_t* x = nullptr;
    uint16_t* y = nullptr;
    int64_t*  t_us = nullptr;
    int8_t*   p = nullptr;
    int       capacity = 0;
  };

  HostStaging& staging_a() { return host_[0]; }
  HostStaging& staging_b() { return host_[1]; }

  // Async: copy SoA host staging (pinned) for n events, clear volume, voxelize into d_vol_f32_,
  // then optional FP16 cast to d_vol_f16_. Non-blocking on `stream`.
  void enqueue(const HostStaging& host_soa, int n,
               int64_t t0_us, int64_t t1_us,
               bool produce_f16,
               cudaStream_t stream);

  [[nodiscard]] float*       device_volume_f32() const noexcept { return d_vol_f32_; }
  [[nodiscard]] __half*      device_volume_f16() const noexcept { return d_vol_f16_; }
  [[nodiscard]] VoxelShape   shape() const noexcept { return shape_; }
  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

  // Rough exclusive VRAM used by this object (bytes).
  [[nodiscard]] size_t device_bytes() const noexcept;

private:
  VoxelShape shape_{};
  int max_events_ = 0;

  float*  d_vol_f32_ = nullptr;
  __half* d_vol_f16_ = nullptr;

  uint16_t* d_x_ = nullptr;
  uint16_t* d_y_ = nullptr;
  int64_t*  d_t_ = nullptr;
  int8_t*   d_p_ = nullptr;

  HostStaging host_[2]{};
  cudaStream_t stream_ = nullptr;
};

// Launch helpers (also used from tests)
void launch_voxelize(const uint16_t* d_x, const uint16_t* d_y,
                     const int64_t* d_t, const int8_t* d_p,
                     int n, int height, int width, int bins,
                     int64_t t0_us, int64_t t1_us,
                     float* d_vol, cudaStream_t stream);

void launch_f32_to_f16(const float* in, __half* out, size_t n, cudaStream_t stream);
void launch_zero_f32(float* d, size_t n, cudaStream_t stream);

} // namespace eventgate
