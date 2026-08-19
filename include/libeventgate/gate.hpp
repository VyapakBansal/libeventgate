#pragma once

#include "types.hpp"

namespace eventgate {

// STATIC iff ||omega|| < thresh (Skog ARE, one sample). Apply the same instance
// to every camera at time t so a stereo pair holds together.
class ImuHardGate {
public:
  explicit ImuHardGate(float gyro_norm_thresh) : thresh_(gyro_norm_thresh) {}

  [[nodiscard]] GateState decide(float gyro_norm) const noexcept {
    return (gyro_norm < thresh_) ? GateState::Static : GateState::Moving;
  }

  [[nodiscard]] GateState decide(const ImuSample& s) const noexcept {
    return decide(s.gyro_norm());
  }

  [[nodiscard]] float threshold() const noexcept { return thresh_; }

private:
  float thresh_;
};

} // namespace eventgate
