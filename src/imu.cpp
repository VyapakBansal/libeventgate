#include "libeventgate/imu.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace eventgate {

void ImuCsv::clear() {
  t_us_.clear();
  gyro_x_.clear();
  gyro_y_.clear();
  gyro_z_.clear();
  accel_x_.clear();
  accel_y_.clear();
  accel_z_.clear();
}

void ImuCsv::reserve(size_t n) {
  t_us_.reserve(n);
  gyro_x_.reserve(n);
  gyro_y_.reserve(n);
  gyro_z_.reserve(n);
  accel_x_.reserve(n);
  accel_y_.reserve(n);
  accel_z_.reserve(n);
}

void ImuCsv::push(const ImuSample& s) {
  t_us_.push_back(s.t_us);
  gyro_x_.push_back(s.gyro[0]);
  gyro_y_.push_back(s.gyro[1]);
  gyro_z_.push_back(s.gyro[2]);
  accel_x_.push_back(s.accel[0]);
  accel_y_.push_back(s.accel[1]);
  accel_z_.push_back(s.accel[2]);
}

ImuCsv::ImuCsv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open IMU CSV: " + path);

  std::string header;
  if (!std::getline(in, header)) throw std::runtime_error("empty IMU CSV: " + path);
  if (header.find("timestamp_us") == std::string::npos ||
      header.find("gyro_x") == std::string::npos) {
    throw std::runtime_error(
        "IMU CSV header must include timestamp_us,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z");
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    for (char& c : line) {
      if (c == ',') c = ' ';
    }
    std::istringstream ss(line);
    ImuSample s;
    if (!(ss >> s.t_us >> s.gyro[0] >> s.gyro[1] >> s.gyro[2] >> s.accel[0] >> s.accel[1] >>
          s.accel[2])) {
      throw std::runtime_error("bad IMU CSV row: " + line);
    }
    push(s);
  }
  if (t_us_.empty()) throw std::runtime_error("IMU CSV has no samples: " + path);
}

ImuSample ImuCsv::at(size_t i) const {
  if (i >= t_us_.size()) throw std::runtime_error("IMU index out of range");
  ImuSample s;
  s.t_us = t_us_[i];
  s.gyro[0] = gyro_x_[i];
  s.gyro[1] = gyro_y_[i];
  s.gyro[2] = gyro_z_[i];
  s.accel[0] = accel_x_[i];
  s.accel[1] = accel_y_[i];
  s.accel[2] = accel_z_[i];
  return s;
}

ImuSample ImuCsv::sample_at(int64_t t_us) const {
  if (t_us_.empty()) throw std::runtime_error("IMU series is empty");
  auto it = std::lower_bound(t_us_.begin(), t_us_.end(), t_us);
  size_t i = 0;
  if (it == t_us_.begin()) {
    i = 0;
  } else if (it == t_us_.end()) {
    i = t_us_.size() - 1;
  } else {
    const size_t hi = static_cast<size_t>(it - t_us_.begin());
    const size_t lo = hi - 1;
    i = (std::llabs(t_us_[hi] - t_us) < std::llabs(t_us_[lo] - t_us)) ? hi : lo;
  }
  return at(i);
}

ImuCsv ImuCsv::make_synthetic(int64_t duration_us, float rate_hz) {
  ImuCsv out;
  const double dt_us = 1e6 / static_cast<double>(rate_hz);
  const int n = static_cast<int>(static_cast<double>(duration_us) / dt_us);
  out.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    ImuSample s;
    s.t_us = static_cast<int64_t>(i * dt_us);
    const float g = (i < n / 2) ? 0.5f : 0.01f;
    s.gyro[0] = g;
    s.gyro[1] = 0.f;
    s.gyro[2] = 0.f;
    s.accel[0] = 0.f;
    s.accel[1] = 0.f;
    s.accel[2] = 9.81f;
    out.push(s);
  }
  return out;
}

void ImuCsv::write_synthetic(const std::string& path, int64_t duration_us, float rate_hz) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write IMU: " + path);
  out << "timestamp_us,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z\n";
  const double dt_us = 1e6 / static_cast<double>(rate_hz);
  const int n = static_cast<int>(static_cast<double>(duration_us) / dt_us);
  for (int i = 0; i < n; ++i) {
    const int64_t t = static_cast<int64_t>(i * dt_us);
    const float g = (i < n / 2) ? 0.5f : 0.01f;
    out << t << ',' << g << ",0,0,0,0,9.81\n";
  }
}

} // namespace eventgate
