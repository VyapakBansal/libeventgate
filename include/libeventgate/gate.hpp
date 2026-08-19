#pragma once

#include "types.hpp"

namespace eventgate {

// Hard gate: STATIC iff ||ω|| < thresh (ARE-style; no radar).
// Pipeline holds a lookback reconstruction while STATIC (same as empty-window HOLD).
// Stereo EVS: construct ONE gate and apply decide() to every camera at the same t.
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
