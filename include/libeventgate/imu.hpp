#pragma once

#include "types.hpp"

#include <string>
#include <vector>

namespace eventgate {

// IMU series: CSV sidecar and/or MCAP /imu topic.
// CSV columns: timestamp_us,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z
// Assumed sorted ascending timestamp_us. Nearest-neighbor lookup O(log n).
class ImuCsv {
public:
  ImuCsv() = default;
  explicit ImuCsv(const std::string& path);

  [[nodiscard]] size_t size() const noexcept { return t_us_.size(); }
  [[nodiscard]] bool empty() const noexcept { return t_us_.empty(); }
  [[nodiscard]] ImuSample sample_at(int64_t t_us) const;
  [[nodiscard]] ImuSample at(size_t i) const;

  void clear();
  void reserve(size_t n);
  void push(const ImuSample& s);

  // Write synthetic IMU (spin then near-static) for smoke tests.
  static void write_synthetic(const std::string& path,
                              int64_t duration_us,
                              float rate_hz = 200.f);

  // Fill series in memory (same pattern as CSV synthetic).
  static ImuCsv make_synthetic(int64_t duration_us, float rate_hz = 200.f);

private:
  std::vector<int64_t> t_us_;
  std::vector<float> gyro_x_, gyro_y_, gyro_z_;
  std::vector<float> accel_x_, accel_y_, accel_z_;
};

} // namespace eventgate
