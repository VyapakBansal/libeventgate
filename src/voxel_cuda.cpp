#include "libeventgate/voxel_cuda.hpp"

#include <stdexcept>
#include <string>

// voxelize kernel declarations live next to the .cu
namespace eventgate {
namespace cuda_kernels {
void voxelize(const uint16_t* d_x, const uint16_t* d_y, const int64_t* d_t, const int8_t* d_p,
              int n, int height, int width, int bins, int64_t t0_us, int64_t t1_us, float* d_vol,
              cudaStream_t stream);
void zero_f32(float* d, size_t n, cudaStream_t stream);
void f32_to_f16(const float* in, __half* out, size_t n, cudaStream_t stream);
} // namespace cuda_kernels
} // namespace eventgate

#include <cuda_fp16.h>
#include <cstring>

namespace eventgate {

VoxelizerCuda::VoxelizerCuda(VoxelShape shape, int max_events_per_window, int /*stream_priority*/)
    : shape_(shape), max_events_(max_events_per_window) {
  cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");

  cuda_check(cudaMalloc(&d_vol_f32_, shape_.bytes_f32()), "cudaMalloc vol f32");
  cuda_check(cudaMalloc(&d_vol_f16_, shape_.bytes_f16()), "cudaMalloc vol f16");

  const size_t ne = static_cast<size_t>(max_events_);
  cuda_check(cudaMalloc(&d_x_, ne * sizeof(uint16_t)), "cudaMalloc d_x");
  cuda_check(cudaMalloc(&d_y_, ne * sizeof(uint16_t)), "cudaMalloc d_y");
  cuda_check(cudaMalloc(&d_t_, ne * sizeof(int64_t)), "cudaMalloc d_t");
  cuda_check(cudaMalloc(&d_p_, ne * sizeof(int8_t)), "cudaMalloc d_p");

  for (int i = 0; i < 2; ++i) {
    host_[i].capacity = max_events_;
    // Pinned host staging → async H2D bandwidth
    cuda_check(cudaHostAlloc(&host_[i].x, ne * sizeof(uint16_t), cudaHostAllocDefault),
               "cudaHostAlloc x");
    cuda_check(cudaHostAlloc(&host_[i].y, ne * sizeof(uint16_t), cudaHostAllocDefault),
               "cudaHostAlloc y");
    cuda_check(cudaHostAlloc(&host_[i].t_us, ne * sizeof(int64_t), cudaHostAllocDefault),
               "cudaHostAlloc t");
    cuda_check(cudaHostAlloc(&host_[i].p, ne * sizeof(int8_t), cudaHostAllocDefault),
               "cudaHostAlloc p");
  }
}

VoxelizerCuda::~VoxelizerCuda() {
  for (int i = 0; i < 2; ++i) {
    if (host_[i].x) cudaFreeHost(host_[i].x);
    if (host_[i].y) cudaFreeHost(host_[i].y);
    if (host_[i].t_us) cudaFreeHost(host_[i].t_us);
    if (host_[i].p) cudaFreeHost(host_[i].p);
  }
  if (d_x_) cudaFree(d_x_);
  if (d_y_) cudaFree(d_y_);
  if (d_t_) cudaFree(d_t_);
  if (d_p_) cudaFree(d_p_);
  if (d_vol_f32_) cudaFree(d_vol_f32_);
  if (d_vol_f16_) cudaFree(d_vol_f16_);
  if (stream_) cudaStreamDestroy(stream_);
}

size_t VoxelizerCuda::device_bytes() const noexcept {
  const size_t ne = static_cast<size_t>(max_events_);
  return shape_.bytes_f32() + shape_.bytes_f16() +
         ne * (sizeof(uint16_t) * 2 + sizeof(int64_t) + sizeof(int8_t));
}

void VoxelizerCuda::enqueue(const HostStaging& host_soa,
                            int n,
                            int64_t t0_us,
                            int64_t t1_us,
                            bool produce_f16,
                            cudaStream_t stream) {
  if (n > max_events_) {
    throw std::runtime_error("VoxelizerCuda: n exceeds max_events_per_window capacity");
  }
  cudaStream_t s = stream ? stream : stream_;

  // Clear volume
  cuda_kernels::zero_f32(d_vol_f32_, shape_.elements(), s);

  if (n > 0) {
    const size_t ne = static_cast<size_t>(n);
    cuda_check(cudaMemcpyAsync(d_x_, host_soa.x, ne * sizeof(uint16_t), cudaMemcpyHostToDevice, s),
               "H2D x");
    cuda_check(cudaMemcpyAsync(d_y_, host_soa.y, ne * sizeof(uint16_t), cudaMemcpyHostToDevice, s),
               "H2D y");
    cuda_check(cudaMemcpyAsync(d_t_, host_soa.t_us, ne * sizeof(int64_t), cudaMemcpyHostToDevice, s),
               "H2D t");
    cuda_check(cudaMemcpyAsync(d_p_, host_soa.p, ne * sizeof(int8_t), cudaMemcpyHostToDevice, s),
               "H2D p");

    cuda_kernels::voxelize(d_x_, d_y_, d_t_, d_p_, n, shape_.height, shape_.width, shape_.bins,
                           t0_us, t1_us, d_vol_f32_, s);
  }

  if (produce_f16) {
    cuda_kernels::f32_to_f16(d_vol_f32_, d_vol_f16_, shape_.elements(), s);
  }
}

void launch_voxelize(const uint16_t* d_x,
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
  cuda_kernels::voxelize(d_x, d_y, d_t, d_p, n, height, width, bins, t0_us, t1_us, d_vol, stream);
}

void launch_f32_to_f16(const float* in, __half* out, size_t n, cudaStream_t stream) {
  cuda_kernels::f32_to_f16(in, out, n, stream);
}

void launch_zero_f32(float* d, size_t n, cudaStream_t stream) {
  cuda_kernels::zero_f32(d, n, stream);
}

} // namespace eventgate
