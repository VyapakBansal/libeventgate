#include "libeventgate/vram.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace eventgate {

VramSnapshot query_vram() {
  VramSnapshot s;
  size_t free_b = 0, total_b = 0;
  const cudaError_t e = cudaMemGetInfo(&free_b, &total_b);
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemGetInfo: ") + cudaGetErrorString(e));
  }
  s.free_bytes = free_b;
  s.total_bytes = total_b;
  s.used_bytes = total_b - free_b;
  return s;
}

VramSnapshot log_vram(const std::string& label, float warn_mib) {
  const auto s = query_vram();
  std::printf("[vram] %s: used=%.0f / %.0f MiB (free=%.0f)\n",
              label.c_str(),
              s.used_mib(),
              s.total_mib(),
              s.free_bytes / (1024.f * 1024.f));
  if (s.used_mib() > warn_mib) {
    std::fprintf(stderr,
                 "[vram] WARN: used %.0f MiB > soft budget %.0f MiB — shrink TRT workspace / force INT8\n",
                 s.used_mib(),
                 warn_mib);
  }
  return s;
}

} // namespace eventgate
