#include "libeventgate/pipeline.hpp"

#include "libeventgate/gate.hpp"
#include "libeventgate/hdf5_events.hpp"
#include "libeventgate/imu.hpp"
#include "libeventgate/keypoints.hpp"
#include "libeventgate/voxel_cuda.hpp"
#include "libeventgate/vram.hpp"

#ifndef LIBEVENTGATE_HAS_TRT
#define LIBEVENTGATE_HAS_TRT 0
#endif
#if LIBEVENTGATE_HAS_TRT
#include "libeventgate/firenet_trt.hpp"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace eventgate {
namespace {

// Proxy "reconstruction" without engine: sum polarity voxels → gray. Enough to
// prove pipeline + empty-window collapse for Day 1–2 smoke. Replaced by TRT.
void proxy_frame_from_voxel(const float* vol, int bins, int H, int W, float* out) {
  const size_t hw = static_cast<size_t>(H) * static_cast<size_t>(W);
  std::memset(out, 0, hw * sizeof(float));
  for (int b = 0; b < bins; ++b) {
    const float* slab = vol + static_cast<size_t>(b) * hw;
    for (size_t i = 0; i < hw; ++i) out[i] += slab[i];
  }
  float mn = out[0], mx = out[0];
  for (size_t i = 1; i < hw; ++i) {
    mn = std::min(mn, out[i]);
    mx = std::max(mx, out[i]);
  }
  const float den = (mx > mn) ? (mx - mn) : 1.f;
  for (size_t i = 0; i < hw; ++i) out[i] = (out[i] - mn) / den;
}

} // namespace

PipelineResult run_phase0(const PipelineConfig& cfg) {
  print_config(cfg);
  fs::create_directories(cfg.out_dir);

  if (cfg.events_h5.empty()) {
    throw std::runtime_error("--events is required");
  }

  log_vram("pipeline_start", cfg.vram_warn_mib);

  EventHdf5 events(cfg.events_h5);
  const auto& meta = events.meta();
  std::cout << "events: n=" << meta.n_events << "  " << meta.width << "x" << meta.height
            << "  t=[" << meta.t_min_us << "," << meta.t_max_us << "] us  sensor=" << meta.sensor
            << "\n";

  std::unique_ptr<ImuCsv> imu;
  std::unique_ptr<ImuHardGate> gate;
  if (!cfg.imu_csv.empty()) {
    imu = std::make_unique<ImuCsv>(cfg.imu_csv);
    std::cout << "imu samples: " << imu->size() << "\n";
  }
  if (cfg.enable_imu_gate) {
    if (!cfg.gyro_thresh_set) {
      throw std::runtime_error("--gate requires --gyro-thresh (set after inspecting real IMU)");
    }
    if (!imu) throw std::runtime_error("--gate requires --imu");
    gate = std::make_unique<ImuHardGate>(cfg.gyro_static_thresh);
  }

  VoxelShape shape{cfg.bins, cfg.height > 0 ? cfg.height : meta.height,
                   cfg.width > 0 ? cfg.width : meta.width};
  // Capacity: worst-case pack for dense 10 ms @ IMX637 — configurable headroom.
  constexpr int kMaxEvents = 1 << 20; // 1M events/window headroom
  VoxelizerCuda voxelizer(shape, kMaxEvents);
  std::cout << "voxel device bytes ≈ " << (voxelizer.device_bytes() / (1024.0 * 1024.0)) << " MiB\n";
  log_vram("after_voxelizer", cfg.vram_warn_mib);

#if LIBEVENTGATE_HAS_TRT
  std::unique_ptr<FireNetTrt> trt;
  if (!cfg.engine_path.empty()) {
    trt = std::make_unique<FireNetTrt>(cfg.engine_path);
    std::cout << "TRT device bytes ≈ " << (trt->device_bytes() / (1024.0 * 1024.0))
              << " MiB  state_io=" << trt->has_state_io()
              << "  state_pairs=" << trt->num_state_pairs()
              << "  voxel_fp16=" << trt->voxel_is_fp16() << "\n";
    if (cfg.enable_imu_gate && !trt->has_state_io()) {
      std::cerr << "[gate] WARN: --gate set but engine has no h_in/h_out — freeze is a no-op. "
                   "Re-export with externalized state.\n";
    }
    log_vram("after_trt_load", cfg.vram_warn_mib);
    // Soft pre-flight: engine bindings + voxel + 512 MB workspace should stay well under 4.5 GB.
    const float est =
        static_cast<float>(trt->device_bytes() + voxelizer.device_bytes()) / (1024.f * 1024.f) +
        512.f;
    if (est > cfg.vram_warn_mib) {
      std::cerr << "[vram] WARN: rough est " << est << " MiB (bindings+voxel+workspace) may "
                   "exceed soft budget " << cfg.vram_warn_mib << " MiB before activations/OpenCV\n";
    } else {
      std::cout << "[vram] preflight est ≈ " << est
                << " MiB (bindings+voxel+workspace) — under soft budget " << cfg.vram_warn_mib
                << " MiB\n";
    }
  }
#else
  if (!cfg.engine_path.empty()) {
    std::cerr << "WARN: built without TensorRT; ignoring --engine\n";
  }
#endif

  // Host output frame (pinned for async D2H)
  float* h_frame = nullptr;
  const size_t frame_elems = static_cast<size_t>(shape.height) * static_cast<size_t>(shape.width);
  cuda_check(cudaHostAlloc(&h_frame, frame_elems * sizeof(float), cudaHostAllocDefault),
             "pin frame");

  // Proxy-only scratch: full-volume D2H. Skipped for TRT (device-resident voxel bind).
  float* d_vol_snapshot = nullptr;
#if LIBEVENTGATE_HAS_TRT
  const bool use_trt = static_cast<bool>(trt);
#else
  const bool use_trt = false;
#endif
  if (!use_trt) {
    cuda_check(cudaMalloc(&d_vol_snapshot, shape.bytes_f32()), "scratch vol");
  }

  const std::string kp_path = cfg.out_dir + "/keypoints.csv";
  const std::string vid_path = cfg.out_dir + "/recon.mp4";
  std::unique_ptr<KeypointCsvWriter> kpw;
  if (cfg.write_keypoint_csv) kpw = std::make_unique<KeypointCsvWriter>(kp_path);
  std::unique_ptr<DemoVideoWriter> vid;
  if (cfg.write_video) {
    const double fps = 1e6 / static_cast<double>(cfg.window_us);
    vid = std::make_unique<DemoVideoWriter>(vid_path, shape.width, shape.height, fps);
    if (!vid->ok()) std::cerr << "WARN: video writer failed to open " << vid_path << "\n";
  }

  PipelineResult result;
  result.keypoint_csv = kp_path;
  result.video_path = vid_path;

  int64_t t_cursor = meta.t_min_us;
  int64_t t_end = meta.t_max_us + cfg.window_us;
  if (meta.n_events == 0) {
    t_cursor = 0;
    t_end = cfg.window_us * 10;
  }

  // Blackout demo tail: empty windows after last event (proxy or real FireNet decay).
  t_end = std::max(t_end, meta.t_max_us + cfg.blackout_tail_us);

  int buf = 0;
  float peak_vram = 0.f;
  const int vram_every = std::max(cfg.vram_log_every, 0);

  while (t_cursor + cfg.window_us <= t_end) {
    const int64_t t0 = t_cursor;
    const int64_t t1 = t_cursor + cfg.window_us;
    t_cursor = t1;

    const auto [i0, i1] = events.index_range(t0, t1);
    const int64_t n64 = i1 - i0;
    const int n = static_cast<int>(std::min<int64_t>(n64, kMaxEvents));

    auto& stage = (buf == 0) ? voxelizer.staging_a() : voxelizer.staging_b();
    buf ^= 1;

    if (n > 0) {
      std::memcpy(stage.x, events.x() + i0, sizeof(uint16_t) * static_cast<size_t>(n));
      std::memcpy(stage.y, events.y() + i0, sizeof(uint16_t) * static_cast<size_t>(n));
      std::memcpy(stage.t_us, events.t() + i0, sizeof(int64_t) * static_cast<size_t>(n));
      std::memcpy(stage.p, events.p() + i0, sizeof(int8_t) * static_cast<size_t>(n));
    }

    GateState gs = GateState::Moving;
    float gnorm = 0.f;
    if (imu) {
      const ImuSample s = imu->sample_at(t0);
      gnorm = s.gyro_norm();
      if (gate) gs = gate->decide(s);
    }

    const bool need_f16 =
#if LIBEVENTGATE_HAS_TRT
        static_cast<bool>(trt) && trt->voxel_is_fp16();
#else
        false;
#endif

    const auto tv0 = std::chrono::steady_clock::now();
    // Device-resident volume only — never pull full voxel to host on TRT path.
    voxelizer.enqueue(stage, n, t0, t1, need_f16, voxelizer.stream());

    const bool freeze = (gs == GateState::Static);

    const auto ti0 = std::chrono::steady_clock::now();
#if LIBEVENTGATE_HAS_TRT
    if (trt) {
      // Zero-copy bind: pass device volume pointer that matches engine voxel dtype.
      const void* vox_ptr = trt->voxel_is_fp16()
                                ? static_cast<const void*>(voxelizer.device_volume_f16())
                                : static_cast<const void*>(voxelizer.device_volume_f32());
      // Single pipeline stream: voxel → TRT → D2H, one sync.
      trt->run_frame(vox_ptr, h_frame, shape.height, shape.width, freeze, voxelizer.stream());
      cuda_check(cudaStreamSynchronize(voxelizer.stream()), "voxel+trt sync");
    } else
#endif
    {
      cuda_check(cudaStreamSynchronize(voxelizer.stream()), "voxel sync");
      cuda_check(cudaMemcpy(d_vol_snapshot, voxelizer.device_volume_f32(), shape.bytes_f32(),
                            cudaMemcpyDeviceToDevice),
                 "snapshot vol");
      std::vector<float> hvol(shape.elements());
      cuda_check(cudaMemcpy(hvol.data(), d_vol_snapshot, shape.bytes_f32(), cudaMemcpyDeviceToHost),
                 "D2H vol");
      proxy_frame_from_voxel(hvol.data(), shape.bins, shape.height, shape.width, h_frame);
      (void)freeze;
    }
    const auto ti1 = std::chrono::steady_clock::now();
    const float voxel_infer_ms =
        std::chrono::duration<float, std::milli>(ti1 - tv0).count();
    const float infer_ms =
        std::chrono::duration<float, std::milli>(ti1 - ti0).count();
    const float voxel_ms = voxel_infer_ms - infer_ms;

    cv::Mat u8 = frame_to_u8(h_frame, shape.height, shape.width);
    const int kpn = count_keypoints(u8, cfg.keypoint_detector, cfg.max_keypoints);
    const double t_s = static_cast<double>(t0 - meta.t_min_us) * 1e-6;
    if (kpw) kpw->write(t_s, kpn);
    if (vid && vid->ok()) vid->write_gray(u8);

    ++result.windows;
    if (gs == GateState::Static) ++result.static_windows;

    const bool log_this =
        (vram_every > 0 && (result.windows % vram_every) == 0);
    if (log_this) {
      auto v = log_vram("window_" + std::to_string(result.windows), cfg.vram_warn_mib);
      peak_vram = std::max(peak_vram, v.used_mib());
      std::cout << "  t=" << t_s << "s n_evt=" << n << " kp=" << kpn
                << " gate=" << (gs == GateState::Static ? "STATIC" : "MOVING")
                << " gyro=" << gnorm << " voxel_ms=" << voxel_ms << " infer_ms=" << infer_ms
                << "\n";
    }
  }

  if (kpw) kpw->close();
  if (vid) vid->release();

  auto v = log_vram("pipeline_end", cfg.vram_warn_mib);
  result.peak_vram_mib = std::max(peak_vram, v.used_mib());

  if (d_vol_snapshot) cudaFree(d_vol_snapshot);
  cudaFreeHost(h_frame);

  std::cout << "done windows=" << result.windows << " static=" << result.static_windows
            << " peak_vram_mib=" << result.peak_vram_mib << "\n"
            << "  keypoints: " << result.keypoint_csv << "\n"
            << "  video:     " << result.video_path << "\n";
  return result;
}

} // namespace eventgate
