#include "libeventgate/pipeline.hpp"
#include "libeventgate/hdf5_events.hpp"
#include "libeventgate/imu.hpp"
#include "libeventgate/config.hpp"
#include "libeventgate/vram.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

// Day-1 smoke: synthetic HDF5 + IMU → CUDA voxel pipeline → keypoint CSV / video.
// No FireNet weights, no TensorRT required.
int main(int argc, char** argv) {
  using namespace eventgate;
  try {
    const std::string out_root =
        (argc > 1) ? argv[1] : std::string("out/smoke");
    fs::create_directories(out_root);

    log_vram("smoke_start");

    const std::string h5 = out_root + "/synthetic_events.h5";
    const std::string imu = out_root + "/synthetic_imu.csv";

    // 2 s total, events only in first 1 s → second half is blackout
    write_synthetic_events_hdf5(h5, kDefaultWidth, kDefaultHeight,
                                /*duration_us=*/2'000'000,
                                /*active_us=*/1'000'000,
                                /*events_per_us=*/5);
    ImuCsv::write_synthetic(imu, 2'000'000, 200.f);

    PipelineConfig cfg;
    cfg.events_h5 = h5;
    cfg.imu_csv = imu;
    cfg.out_dir = out_root + "/run";
    cfg.window_us = kDefaultWindowUs;
    cfg.bins = kDefaultBins;
    cfg.width = kDefaultWidth;
    cfg.height = kDefaultHeight;
    cfg.enable_imu_gate = true;
    cfg.gyro_static_thresh = 0.05f;
    cfg.gyro_thresh_set = true;
    cfg.write_video = true;

    const auto result = run_phase0(cfg);
    std::cout << "SMOKE OK peak_vram_mib=" << result.peak_vram_mib << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "SMOKE FAIL: " << e.what() << "\n";
    return 1;
  }
}
