#include "voxelize.cuh"

#include <cuda_fp16.h>
#include <cstdio>

namespace eventgate {
namespace cuda_kernels {
namespace {

// ---- helpers ----
__device__ __forceinline__ void atomic_add_vol(float* vol, int bins, int H, int W,
                                               int b, int y, int x, float v) {
  if ((unsigned)b >= (unsigned)bins) return;
  if ((unsigned)y >= (unsigned)H) return;
  if ((unsigned)x >= (unsigned)W) return;
  const size_t idx = (static_cast<size_t>(b) * static_cast<size_t>(H) + static_cast<size_t>(y)) *
                         static_cast<size_t>(W) +
                     static_cast<size_t>(x);
  atomicAdd(vol + idx, v);
}

// One event → bilinear split across two adjacent temporal bins.
__global__ void k_voxelize(const uint16_t* __restrict__ x,
                           const uint16_t* __restrict__ y,
                           const int64_t*  __restrict__ t,
                           const int8_t*   __restrict__ p,
                           int n,
                           int H,
                           int W,
                           int bins,
                           int64_t t0_us,
                           double inv_dt_bins, // bins / (t1-t0)
                           float* __restrict__ vol) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const int xi = static_cast<int>(x[i]);
  const int yi = static_cast<int>(y[i]);
  if ((unsigned)xi >= (unsigned)W || (unsigned)yi >= (unsigned)H) return;

  const double tn = (static_cast<double>(t[i] - t0_us) * inv_dt_bins);
  // clamp to [0, bins)
  double tc = tn;
  if (tc < 0.0) tc = 0.0;
  if (tc >= static_cast<double>(bins)) tc = static_cast<double>(bins) - 1e-6;

  const int b0 = static_cast<int>(tc);
  const int b1 = min(b0 + 1, bins - 1);
  const float w1 = static_cast<float>(tc - static_cast<double>(b0));
  const float w0 = 1.f - w1;
  const float s = static_cast<float>(p[i]); // ±1

  atomic_add_vol(vol, bins, H, W, b0, yi, xi, s * w0);
  if (b1 != b0) {
    atomic_add_vol(vol, bins, H, W, b1, yi, xi, s * w1);
  }
}

__global__ void k_zero(float* d, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) d[i] = 0.f;
}

__global__ void k_f32_to_f16(const float* __restrict__ in, __half* __restrict__ out, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

} // namespace

void voxelize(const uint16_t* d_x,
              const uint16_t* d_y,
              const int64_t* d_t,
              const int8_t* d_p,
              int n,
              int height,
              int width,
              int bins,
              int64_t t0_us,
              int64_t t1_us,
              float* d_vol,
              cudaStream_t stream) {
  if (n <= 0) return;
  const int64_t dt = (t1_us > t0_us) ? (t1_us - t0_us) : 1;
  const double inv = static_cast<double>(bins) / static_cast<double>(dt);

  constexpr int TPB = 256;
  const int blocks = (n + TPB - 1) / TPB;
  k_voxelize<<<blocks, TPB, 0, stream>>>(d_x, d_y, d_t, d_p, n, height, width, bins, t0_us, inv,
                                         d_vol);
}

void zero_f32(float* d, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  constexpr int TPB = 256;
  const int blocks = static_cast<int>((n + TPB - 1) / TPB);
  k_zero<<<blocks, TPB, 0, stream>>>(d, n);
}

void f32_to_f16(const float* in, __half* out, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  constexpr int TPB = 256;
  const int blocks = static_cast<int>((n + TPB - 1) / TPB);
  k_f32_to_f16<<<blocks, TPB, 0, stream>>>(in, out, n);
}

} // namespace cuda_kernels
} // namespace eventgate
