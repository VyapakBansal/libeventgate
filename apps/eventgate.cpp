#include "libeventgate/config.hpp"
#include "libeventgate/pipeline.hpp"

#include <iostream>

int main(int argc, char** argv) {
  using namespace eventgate;
  try {
    PipelineConfig cfg = parse_cli(argc, argv);
    if (cfg.events_h5.empty()) {
      std::cerr << "--events / --events-left required\n";
      return 2;
    }
    if (cfg.events_h5_right.empty()) {
      run_pipeline(cfg);
      return 0;
    }
    // One engine at a time: left then right, same IMU timestamps and gyro threshold.
    const std::string base = cfg.out_dir;
    PipelineConfig left = cfg;
    left.events_h5_right.clear();
    left.out_dir = base + "/left";
    std::cout << "=== stereo left ===\n";
    run_pipeline(left);

    PipelineConfig right = cfg;
    right.events_h5 = cfg.events_h5_right;
    right.events_h5_right.clear();
    right.out_dir = base + "/right";
    std::cout << "=== stereo right (same IMU/gate) ===\n";
    run_pipeline(right);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
