#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace eventgate {
namespace cuda_kernels {

// High-throughput voxelization:
//  - SoA inputs for coalesced loads
//  - FP32 atomicAdd into CHW volume (bin-major)
//  - bilinear weight on time only (space is hard grid sample — event coords are discrete)
//
// Index: ((b * H) + y) * W + x
void voxelize(const uint16_t* __restrict__ d_x,
              const uint16_t* __restrict__ d_y,
              const int64_t*  __restrict__ d_t,
              const int8_t*   __restrict__ d_p,
              int n,
              int height,
              int width,
              int bins,
              int64_t t0_us,
              int64_t t1_us,
              float* __restrict__ d_vol,
              cudaStream_t stream);

void zero_f32(float* d, size_t n, cudaStream_t stream);
void f32_to_f16(const float* in, __half* out, size_t n, cudaStream_t stream);

} // namespace cuda_kernels
} // namespace eventgate
