#pragma once

#include <cstddef>
#include <string>

namespace eventgate {

struct VramSnapshot {
  size_t free_bytes  = 0;
  size_t total_bytes = 0;
  size_t used_bytes  = 0;

  [[nodiscard]] float used_mib()  const noexcept { return used_bytes  / (1024.f * 1024.f); }
  [[nodiscard]] float total_mib() const noexcept { return total_bytes / (1024.f * 1024.f); }
};

VramSnapshot query_vram();
// Prints and returns snapshot. Warns if used_mib > warn_mib.
VramSnapshot log_vram(const std::string& label, float warn_mib = 4500.f);

} // namespace eventgate
