#include "libeventgate/config.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace eventgate {
namespace {

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --events PATH        Event HDF5 (left / mono)\n"
      << "  --events-left PATH   Alias for --events\n"
      << "  --events-right PATH  Optional second EVS HDF5; same IMU gate, sequential pass\n"
      << "  --imu PATH           IMU CSV sidecar (shared across left/right)\n"
      << "  --engine PATH        TensorRT engine (optional for voxel-only / proxy recon)\n"
      << "  --onnx PATH          ONNX (for build_engine)\n"
      << "  --out DIR            Output directory (default: out)\n"
      << "  --window-ms N        Voxel window ms (default: 10)\n"
      << "  --bins N             Temporal bins (default: 5)\n"
      << "  --gyro-thresh X      ||gyro|| static threshold (required with --gate)\n"
      << "  --gate               Enable IMU-only hard gate (needs state I/O engine)\n"
      << "  --hold-lookback-s X     Fade-horizon for HOLD latch (default: 1)\n"
      << "  --hold-min-static-s X   Latch only if STATIC lasts at least X s (default: 1)\n"
      << "  --hold-release-s X      Unfreeze only after X s of MOVING (default: 0.25)\n"
      << "  --detector ORB|FAST  Keypoint detector (default: ORB)\n"
      << "  --no-video           Skip video writer\n"
      << "  --blackout-tail-s X  Empty windows after last event (default: 2)\n"
      << "  --vram-every N       Log VRAM every N windows (default: 50)\n"
      << "  --workspace-mb N     TRT workspace cap (default: 512)\n"
      << "  --fp16               Prefer FP16\n"
      << "  --int8 / --no-int8   INT8 preference (default: on; needs calibrator later)\n"
      << "  --help\n";
}

} // namespace

PipelineConfig parse_cli(int argc, char** argv) {
  PipelineConfig c;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (a == "--events" || a == "--events-left") {
      c.events_h5 = need(a.c_str());
    } else if (a == "--events-right") {
      c.events_h5_right = need("--events-right");
    } else if (a == "--imu") {
      c.imu_csv = need("--imu");
    } else if (a == "--engine") {
      c.engine_path = need("--engine");
    } else if (a == "--onnx") {
      c.onnx_path = need("--onnx");
    } else if (a == "--out") {
      c.out_dir = need("--out");
    } else if (a == "--window-ms") {
      c.window_us = static_cast<int>(std::stof(need("--window-ms")) * 1000.f);
    } else if (a == "--bins") {
      c.bins = std::stoi(need("--bins"));
    } else if (a == "--gyro-thresh") {
      c.gyro_static_thresh = std::stof(need("--gyro-thresh"));
      c.gyro_thresh_set = true;
    } else if (a == "--gate") {
      c.enable_imu_gate = true;
    } else if (a == "--detector") {
      c.keypoint_detector = need("--detector");
    } else if (a == "--no-video") {
      c.write_video = false;
    } else if (a == "--workspace-mb") {
      c.trt_workspace_bytes =
          static_cast<size_t>(std::stoull(need("--workspace-mb"))) * 1024ull * 1024ull;
    } else if (a == "--int8") {
      c.prefer_int8 = true;
    } else if (a == "--no-int8") {
      c.prefer_int8 = false;
    } else if (a == "--fp16") {
      c.prefer_fp16 = true;
    } else if (a == "--hold-lookback-s") {
      c.hold_lookback_us =
          static_cast<int64_t>(std::stof(need("--hold-lookback-s")) * 1e6f);
    } else if (a == "--hold-min-static-s") {
      c.hold_min_static_us =
          static_cast<int64_t>(std::stof(need("--hold-min-static-s")) * 1e6f);
    } else if (a == "--hold-release-s") {
      c.hold_release_us =
          static_cast<int64_t>(std::stof(need("--hold-release-s")) * 1e6f);
    } else if (a == "--blackout-tail-s") {
      c.blackout_tail_us =
          static_cast<int64_t>(std::stof(need("--blackout-tail-s")) * 1e6f);
    } else if (a == "--vram-every") {
      c.vram_log_every = std::stoi(need("--vram-every"));
    } else if (a == "--width") {
      c.width = std::stoi(need("--width"));
    } else if (a == "--height") {
      c.height = std::stoi(need("--height"));
    } else {
      std::cerr << "unknown arg: " << a << "\n";
      usage(argv[0]);
      std::exit(2);
    }
  }
  return c;
}

void print_config(const PipelineConfig& c) {
  std::cout << "=== PipelineConfig ===\n"
            << "  events:   " << c.events_h5 << "\n"
            << "  events_R: " << (c.events_h5_right.empty() ? "(none)" : c.events_h5_right) << "\n"
            << "  imu:      " << c.imu_csv << "\n"
            << "  engine:   " << c.engine_path << "\n"
            << "  out:      " << c.out_dir << "\n"
            << "  frame:    " << c.width << "x" << c.height
            << "  bins=" << c.bins << "  window_us=" << c.window_us << "\n"
            << "  gate:     " << (c.enable_imu_gate ? "IMU_HARD" : "off")
            << "  thresh=" << (c.gyro_thresh_set ? std::to_string(c.gyro_static_thresh) : "UNSET")
            << "  lookback_s=" << (c.hold_lookback_us * 1e-6)
            << "  min_static_s=" << (c.hold_min_static_us * 1e-6)
            << "  release_s=" << (c.hold_release_us * 1e-6)
            << "\n"
            << "  TRT:      workspace_mb=" << (c.trt_workspace_bytes / (1024 * 1024))
            << "  int8=" << c.prefer_int8 << "  fp16=" << c.prefer_fp16 << "\n"
            << "  blackout: tail_s=" << (c.blackout_tail_us * 1e-6) << "\n";
}

} // namespace eventgate
