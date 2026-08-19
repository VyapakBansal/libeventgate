#pragma once

#include "types.hpp"

#include <string>
#include <vector>
#include <cstdint>
#include <utility>

namespace eventgate {

// Loads /events/{x,y,t_us,p}. File polarity is 0/1; converted to +/-1 in memory.
class EventHdf5 {
public:
  explicit EventHdf5(const std::string& path);
  ~EventHdf5();

  EventHdf5(const EventHdf5&) = delete;
  EventHdf5& operator=(const EventHdf5&) = delete;

  [[nodiscard]] const EventStreamMeta& meta() const noexcept { return meta_; }
  [[nodiscard]] int64_t size() const noexcept { return n_; }

  // Events are sorted by t_us. Half-open index range [i0, i1) in [t0, t1).
  [[nodiscard]] std::pair<int64_t, int64_t> index_range(int64_t t0_us, int64_t t1_us) const;

  [[nodiscard]] const uint16_t* x()   const noexcept { return x_; }
  [[nodiscard]] const uint16_t* y()   const noexcept { return y_; }
  [[nodiscard]] const int64_t*  t()   const noexcept { return t_; }
  [[nodiscard]] const int8_t*   p()   const noexcept { return p_; }

  [[nodiscard]] EventSoA view(int64_t i0, int64_t i1) const;

private:
  EventStreamMeta meta_{};
  int64_t n_ = 0;
  uint16_t* x_ = nullptr;
  uint16_t* y_ = nullptr;
  int64_t*  t_ = nullptr;
  int8_t*   p_ = nullptr;
};

void write_synthetic_events_hdf5(const std::string& path,
                                 int width, int height,
                                 int64_t duration_us,
                                 int64_t active_us,
                                 int events_per_active_us);

} // namespace eventgate
