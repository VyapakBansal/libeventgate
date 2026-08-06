#include "libeventgate/config.hpp"
#include "libeventgate/pipeline.hpp"

#include <iostream>

int main(int argc, char** argv) {
  using namespace eventgate;
  try {
    PipelineConfig cfg = parse_cli(argc, argv);
    if (cfg.events_h5.empty()) {
      std::cerr << "--events required\n";
      return 2;
    }
    run_phase0(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
