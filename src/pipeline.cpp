#include "libeventgate/pipeline.hpp"

#include "libeventgate/gate.hpp"
#include "libeventgate/hdf5_events.hpp"
#include "libeventgate/imu.hpp"
#include "libeventgate/keypoints.hpp"
#include "libeventgate/voxel_cuda.hpp"
#include "libeventgate/vram.hpp"

#include <opencv2/core.hpp>

#ifndef LIBEVENTGATE_HAS_TRT
#define LIBEVENTGATE_HAS_TRT 0
#endif
#if LIBEVENTGATE_HAS_TRT
#include "libeventgate/firenet_trt.hpp"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace eventgate {
namespace {

struct ReconSnap {
  cv::Mat u8;
  int n_events = 0;
};

// Last useful canvas in the fade horizon: most recent window whose event count
// is at least half the peak in the buffer. Skips the starved tail without
// jumping a full second into a different pose.
cv::Mat latch_hold_frame(const std::deque<ReconSnap>& hist, int height, int width,
                         const float* h_frame) {
  if (hist.empty()) {
    return frame_to_u8(h_frame, height, width);
  }
  int n_star = 0;
  for (const auto& s : hist) n_star = std::max(n_star, s.n_events);
  const int thresh = std::max(1, n_star / 2);
  for (auto it = hist.rbegin(); it != hist.rend(); ++it) {
    if (it->n_events >= thresh) return it->u8.clone();
  }
  return hist.front().u8.clone();
}

static cv::Mat blend_u8(const cv::Mat& latch, const cv::Mat& live, float alpha) {
  cv::Mat out;
  cv::addWeighted(latch, 1.f - alpha, live, alpha, 0, out);
  return out;
}

// Sum polarity voxels to a grayscale proxy when no TensorRT engine is loaded.
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

PipelineResult run_pipeline(const PipelineConfig& cfg) {
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
      throw std::runtime_error("--gate requires --gyro-thresh");
    }
    if (!imu) throw std::runtime_error("--gate requires --imu");
    gate = std::make_unique<ImuHardGate>(cfg.gyro_static_thresh);
  }

  VoxelShape shape{cfg.bins, cfg.height > 0 ? cfg.height : meta.height,
                   cfg.width > 0 ? cfg.width : meta.width};
  constexpr int kMaxEvents = 1 << 20; // 1M events/window headroom for dense 10 ms IMX637
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
      std::cerr << "[gate] WARN: --gate set but engine has no h_in/h_out. "
                   "STATIC skip-infer still holds the last canvas, but MOVING FireNet "
                   "will not keep recurrent state. Re-export with externalized state.\n";
    }
    log_vram("after_trt_load", cfg.vram_warn_mib);
    const float est =
        static_cast<float>(trt->device_bytes() + voxelizer.device_bytes()) / (1024.f * 1024.f) +
        512.f;
    if (est > cfg.vram_warn_mib) {
      std::cerr << "[vram] WARN: rough est " << est << " MiB (bindings+voxel+workspace) may "
                   "exceed soft budget " << cfg.vram_warn_mib << " MiB before activations/OpenCV\n";
    } else {
      std::cout << "[vram] preflight est ≈ " << est
                << " MiB (bindings+voxel+workspace); budget " << cfg.vram_warn_mib
                << " MiB\n";
    }
  }
#else
  if (!cfg.engine_path.empty()) {
    std::cerr << "WARN: built without TensorRT; ignoring --engine\n";
  }
#endif

  // Pinned host frame for async D2H.
  float* h_frame = nullptr;
  const size_t frame_elems = static_cast<size_t>(shape.height) * static_cast<size_t>(shape.width);
  cuda_check(cudaHostAlloc(&h_frame, frame_elems * sizeof(float), cudaHostAllocDefault),
             "pin frame");

  // Full-volume D2H only when there is no TRT bind (proxy path).
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
  std::unique_ptr<ReconVideoWriter> vid;
  if (cfg.write_video) {
    const double fps = 1e6 / static_cast<double>(cfg.window_us);
    vid = std::make_unique<ReconVideoWriter>(vid_path, shape.width, shape.height, fps);
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

  t_end = std::max(t_end, meta.t_max_us + cfg.blackout_tail_us);

  int buf = 0;
  float peak_vram = 0.f;
  const int vram_every = std::max(cfg.vram_log_every, 0);
  bool have_frame = false;
  int held_windows = 0;
  const int lookback_n = std::max(
      1, static_cast<int>((cfg.hold_lookback_us + cfg.window_us - 1) / cfg.window_us));
  const int release_n = std::max(
      1, static_cast<int>((cfg.hold_release_us + cfg.window_us - 1) / cfg.window_us));
  const int min_static_n = std::max(
      1, static_cast<int>((cfg.hold_min_static_us + cfg.window_us - 1) / cfg.window_us));
  std::deque<ReconSnap> recon_hist;
  cv::Mat latched;
  cv::Mat static_candidate;  // lookback snapshot at STATIC onset, before debounce wait
  bool in_hold = false;
  int moving_streak = 0;
  int static_streak = 0;
  int n_prev = 0;
  int n_prev2 = 0;
  double com_x_prev = 0.;
  double com_y_prev = 0.;
  bool have_com_prev = false;
  bool tedg_armed = false;
  cv::Mat release_latch;
  int blend_remaining = 0;
  const bool ext =
      cfg.enable_imu_gate && !cfg.ablate_freeze_h &&
      (cfg.enable_cmdg || cfg.enable_tedg || cfg.enable_srb);

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

    double sum_x = 0.;
    double sum_y = 0.;
    if (n > 0) {
      for (int ei = 0; ei < n; ++ei) {
        sum_x += stage.x[ei];
        sum_y += stage.y[ei];
      }
    }

    GateState gs = GateState::Moving;
    float gnorm = 0.f;
    if (imu) {
      const ImuSample s = imu->sample_at(t0);
      gnorm = s.gyro_norm();
      if (gate) gs = gate->decide(s);
    }

    const bool gyro_static = static_cast<bool>(gate) && gs == GateState::Static;
    bool com_veto = false;
    if (ext && cfg.enable_cmdg && n >= cfg.com_min_events && have_com_prev) {
      const double xc = sum_x / static_cast<double>(n);
      const double yc = sum_y / static_cast<double>(n);
      const double drift =
          std::hypot(xc - com_x_prev, yc - com_y_prev);
      if (drift > static_cast<double>(cfg.com_drift_thresh_px)) com_veto = true;
    }
    if (n >= cfg.com_min_events) {
      com_x_prev = sum_x / static_cast<double>(n);
      com_y_prev = sum_y / static_cast<double>(n);
      have_com_prev = true;
    }

    const bool decay =
        ext && cfg.enable_tedg && n >= cfg.tedg_min_events &&
        n_prev >= cfg.tedg_min_events && n_prev2 >= cfg.tedg_min_events &&
        n < static_cast<int>(cfg.tedg_alpha * static_cast<float>(n_prev)) &&
        n_prev < static_cast<int>(cfg.tedg_alpha * static_cast<float>(n_prev2));

    if (decay && have_frame && static_candidate.empty()) {
      static_candidate =
          latch_hold_frame(recon_hist, shape.height, shape.width, h_frame);
      tedg_armed = true;
    }

    // Hold when the window is empty, or STATIC has lasted hold_min_static_s.
    // Snapshot the fade-horizon latch at STATIC onset so the debounce wait does
    // not fill the buffer with dying reconstructions.
    const bool want_static = gyro_static && !com_veto;
    if (want_static) {
      if (static_streak == 0 && have_frame && static_candidate.empty()) {
        static_candidate =
            latch_hold_frame(recon_hist, shape.height, shape.width, h_frame);
        tedg_armed = false;
      }
      ++static_streak;
      moving_streak = 0;
    } else {
      static_streak = 0;
      const bool rate_recovered =
          n > static_cast<int>(cfg.tedg_alpha * static_cast<float>(std::max(n_prev, 1)));
      if (static_candidate.empty() || !tedg_armed || rate_recovered) {
        static_candidate.release();
        tedg_armed = false;
      }
      if (n == 0) {
        moving_streak = 0;
      } else {
        ++moving_streak;
      }
    }
    const bool static_confirmed = want_static && static_streak >= min_static_n;
    // Freeze-h ablation still infers during STATIC (discards h_out). Latch skips f.
    const bool hold =
        have_frame &&
        (n == 0 ||
         (!cfg.ablate_freeze_h &&
          (static_confirmed || (in_hold && moving_streak < release_n))));
    const bool freeze_h [[maybe_unused]] = cfg.ablate_freeze_h && want_static;

    const bool need_f16 =
#if LIBEVENTGATE_HAS_TRT
        static_cast<bool>(trt) && trt->voxel_is_fp16();
#else
        false;
#endif

    const auto tv0 = std::chrono::steady_clock::now();
    float voxel_ms = 0.f;
    float infer_ms = 0.f;
    cv::Mat u8;

    if (hold) {
      ++held_windows;
      if (latched.empty()) {
        latched = static_candidate.empty()
                      ? latch_hold_frame(recon_hist, shape.height, shape.width, h_frame)
                      : static_candidate.clone();
      }
      u8 = latched;
      in_hold = true;
    } else {
      if (in_hold && ext && cfg.enable_srb && cfg.release_blend_n > 0 && !latched.empty()) {
        release_latch = latched.clone();
        blend_remaining = cfg.release_blend_n;
      }
      // Device-resident volume only, never pull full voxel to host on TRT path.
      voxelizer.enqueue(stage, n, t0, t1, need_f16, voxelizer.stream());

      const auto ti0 = std::chrono::steady_clock::now();
#if LIBEVENTGATE_HAS_TRT
      if (trt) {
        const void* vox_ptr = trt->voxel_is_fp16()
                                  ? static_cast<const void*>(voxelizer.device_volume_f16())
                                  : static_cast<const void*>(voxelizer.device_volume_f32());
        trt->run_frame(vox_ptr, h_frame, shape.height, shape.width,
                       /*freeze_state=*/freeze_h, voxelizer.stream());
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
      }
      const auto ti1 = std::chrono::steady_clock::now();
      const float voxel_infer_ms =
          std::chrono::duration<float, std::milli>(ti1 - tv0).count();
      infer_ms = std::chrono::duration<float, std::milli>(ti1 - ti0).count();
      voxel_ms = voxel_infer_ms - infer_ms;
      cv::Mat u8_new = frame_to_u8(h_frame, shape.height, shape.width);
      if (ext && cfg.enable_srb && blend_remaining > 0 && !release_latch.empty()) {
        const int step = cfg.release_blend_n - blend_remaining + 1;
        const float alpha =
            static_cast<float>(step) / static_cast<float>(cfg.release_blend_n);
        u8 = blend_u8(release_latch, u8_new, alpha);
        --blend_remaining;
      } else {
        u8 = u8_new;
      }
      recon_hist.push_back(ReconSnap{u8.clone(), n});
      while (static_cast<int>(recon_hist.size()) > lookback_n) recon_hist.pop_front();
      latched.release();
      in_hold = false;
      have_frame = true;
    }
    n_prev2 = n_prev;
    n_prev = n;
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
                << (com_veto ? " COM_VETO" : "")
                << (hold ? " HOLD" : "")
                << (blend_remaining > 0 ? " SRB" : "")
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
            << " held=" << held_windows << " peak_vram_mib=" << result.peak_vram_mib << "\n"
            << "  keypoints: " << result.keypoint_csv << "\n"
            << "  video:     " << result.video_path << "\n";
  return result;
}

} // namespace eventgate
