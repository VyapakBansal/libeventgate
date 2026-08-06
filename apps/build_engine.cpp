#include "libeventgate/config.hpp"
#include "libeventgate/firenet_trt.hpp"

#include <iostream>

// ONNX → TensorRT engine. Workspace hard-capped (default 512 MB) for 6 GB card.
int main(int argc, char** argv) {
  using namespace eventgate;
  try {
    PipelineConfig cfg = parse_cli(argc, argv);
    if (cfg.onnx_path.empty()) {
      std::cerr << "--onnx required\n";
      return 2;
    }
    std::string out = cfg.engine_path;
    if (out.empty()) out = "engines/firenet.engine";

    std::string err;
    if (!build_engine_from_onnx(cfg.onnx_path, out, cfg, &err)) {
      std::cerr << "build failed: " << err << "\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
