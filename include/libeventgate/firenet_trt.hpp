#pragma once

#include "config.hpp"
#include "types.hpp"

#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace eventgate {

// Build serialized engine from ONNX (offline via apps/build_engine).
// Caps workspace to cfg.trt_workspace_bytes (6 GB cards OOM if this is large).
bool build_engine_from_onnx(const std::string& onnx_path,
                            const std::string& engine_out,
                            const PipelineConfig& cfg,
                            std::string* err);

// Runtime FireNet TRT context.
// freeze_state=true → skip recurrent hidden write-back (keep h_{t-1}).
// Requires ONNX with h_in_*/h_out_* tensors (see python/export_firenet_onnx.py).
// Without state I/O, freeze is a no-op with a one-time warning.
class FireNetTrt {
public:
  explicit FireNetTrt(const std::string& engine_path);
  ~FireNetTrt();

  FireNetTrt(const FireNetTrt&) = delete;
  FireNetTrt& operator=(const FireNetTrt&) = delete;

  // voxel_device: NCHW volume already on device. dtype must match voxel_is_fp16().
  // Prefer VoxelizerCuda::device_volume_f16() when voxel_is_fp16(), else f32.
  // out_frame_host: pinned host float[height*width] (NCHW frame, channel 0).
  void run_frame(const void* voxel_device,
                 float* out_frame_host,
                 int height,
                 int width,
                 bool freeze_state,
                 cudaStream_t stream);

  [[nodiscard]] size_t device_bytes() const noexcept { return device_bytes_; }
  [[nodiscard]] bool has_state_io() const noexcept { return has_state_; }
  // True if the engine's voxel input is FP16 (bind f16 volume, skip host cast).
  [[nodiscard]] bool voxel_is_fp16() const noexcept { return voxel_fp16_; }
  [[nodiscard]] size_t voxel_bytes() const noexcept { return voxel_bytes_; }
  [[nodiscard]] int num_state_pairs() const noexcept { return n_state_pairs_; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  size_t device_bytes_ = 0;
  bool has_state_ = false;
  bool voxel_fp16_ = false;
  size_t voxel_bytes_ = 0;
  int n_state_pairs_ = 0;
};

} // namespace eventgate
