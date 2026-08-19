#pragma once

// Optional MCAP event/IMU container. Not wired into the eventgate CLI; HDF5 is
// the supported interchange. Wire layout is little-endian SoA:
//
// EventPacket:
//   uint32 magic       = 0x4B504745  ('EGPK')
//   uint16 version     = 1
//   uint16 flags       = 0
//   uint32 n_events
//   uint16 width
//   uint16 height
//   uint16 x[n]
//   uint16 y[n]
//   int64  t_us[n]
//   uint8  p[n]        // 0/1 on disk, +/-1 in memory
//
// ImuSample (one message = one sample):
//   int64 t_us
//   float gyro_x, gyro_y, gyro_z
//   float accel_x, accel_y, accel_z


#include "types.hpp"
#include "imu.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace eventgate {

inline constexpr const char* kMcapTopicEvents   = "/event_camera/events";
inline constexpr const char* kMcapTopicImu      = "/imu";
inline constexpr const char* kMcapSchemaEvents  = "eventgate.EventPacket";
inline constexpr const char* kMcapSchemaImu     = "eventgate.ImuSample";
inline constexpr const char* kMcapMetadataName  = "eventgate";
inline constexpr uint32_t    kEventPacketMagic  = 0x4B504745u; // 'EGPK' LE
inline constexpr uint16_t    kEventPacketVersion = 1;

// Full-stream load into host SoA (same contract as EventHdf5).
class EventMcap {
public:
  explicit EventMcap(const std::string& path);
  ~EventMcap();

  EventMcap(const EventMcap&) = delete;
  EventMcap& operator=(const EventMcap&) = delete;

  [[nodiscard]] const EventStreamMeta& meta() const noexcept { return meta_; }
  [[nodiscard]] int64_t size() const noexcept { return n_; }

  [[nodiscard]] std::pair<int64_t, int64_t> index_range(int64_t t0_us, int64_t t1_us) const;
  [[nodiscard]] const uint16_t* x() const noexcept { return x_; }
  [[nodiscard]] const uint16_t* y() const noexcept { return y_; }
  [[nodiscard]] const int64_t*  t() const noexcept { return t_; }
  [[nodiscard]] const int8_t*   p() const noexcept { return p_; }
  [[nodiscard]] EventSoA view(int64_t i0, int64_t i1) const;

  // IMU from /imu if present; otherwise load a sidecar CSV.
  [[nodiscard]] bool has_imu() const noexcept { return imu_.size() > 0; }
  [[nodiscard]] const ImuCsv& imu() const noexcept { return imu_; }

private:
  EventStreamMeta meta_{};
  int64_t n_ = 0;
  uint16_t* x_ = nullptr;
  uint16_t* y_ = nullptr;
  int64_t*  t_ = nullptr;
  int8_t*   p_ = nullptr;
  ImuCsv imu_;
};

// Synthetic events plus optional IMU in one MCAP.
void write_synthetic_mcap(const std::string& path,
                          int width,
                          int height,
                          int64_t duration_us,
                          int64_t active_us,
                          int events_per_active_us,
                          bool with_imu = true);

// Rewrite an in-memory event batch set (+ optional IMU samples) to MCAP.
// events_p_file: polarity 0/1 as stored on disk (not ±1).
void write_events_mcap(const std::string& path,
                       int width,
                       int height,
                       const std::string& sensor,
                       const uint16_t* x,
                       const uint16_t* y,
                       const int64_t* t_us,
                       const uint8_t* p01,
                       int64_t n_events,
                       const std::vector<ImuSample>* imu = nullptr,
                       int events_per_packet = 65536);

void convert_hdf5_to_mcap(const std::string& h5_path,
                          const std::string& mcap_path,
                          const std::string& imu_csv_path = {});

} // namespace eventgate
