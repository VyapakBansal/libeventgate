// Single TUs that pulls in the header-only MCAP library implementation.
#define MCAP_IMPLEMENTATION
#include <mcap/reader.hpp>
#include <mcap/writer.hpp>

#include "libeventgate/mcap_io.hpp"
#include "libeventgate/hdf5_events.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace eventgate {
namespace {

[[noreturn]] void die(const std::string& m) { throw std::runtime_error(m); }

// JSON Schema blobs embedded in MCAP Schema records (document binary layout).
constexpr const char* kEventPacketSchemaJson = R"({
  "title": "eventgate.EventPacket",
  "description": "Little-endian SoA event batch. Magic 0x4B504745 ('EGPK'), version 1.",
  "type": "object",
  "properties": {
    "magic": {"type": "integer", "const": 1263552325},
    "version": {"type": "integer", "const": 1},
    "flags": {"type": "integer"},
    "n_events": {"type": "integer"},
    "width": {"type": "integer"},
    "height": {"type": "integer"},
    "x": {"type": "array", "items": {"type": "integer"}},
    "y": {"type": "array", "items": {"type": "integer"}},
    "t_us": {"type": "array", "items": {"type": "integer"}},
    "p": {"type": "array", "items": {"type": "integer"}, "description": "0 or 1 polarity"}
  }
})";

constexpr const char* kImuSampleSchemaJson = R"({
  "title": "eventgate.ImuSample",
  "description": "Little-endian single IMU sample (SI units; gyro rad/s, accel m/s^2).",
  "type": "object",
  "properties": {
    "t_us": {"type": "integer"},
    "gyro_x": {"type": "number"},
    "gyro_y": {"type": "number"},
    "gyro_z": {"type": "number"},
    "accel_x": {"type": "number"},
    "accel_y": {"type": "number"},
    "accel_z": {"type": "number"}
  }
})";

inline void write_le_u16(std::vector<std::byte>& buf, uint16_t v) {
  buf.push_back(std::byte{static_cast<uint8_t>(v & 0xff)});
  buf.push_back(std::byte{static_cast<uint8_t>((v >> 8) & 0xff)});
}
inline void write_le_u32(std::vector<std::byte>& buf, uint32_t v) {
  for (int i = 0; i < 4; ++i)
    buf.push_back(std::byte{static_cast<uint8_t>((v >> (8 * i)) & 0xff)});
}
inline void write_le_i64(std::vector<std::byte>& buf, int64_t v) {
  const uint64_t u = static_cast<uint64_t>(v);
  for (int i = 0; i < 8; ++i)
    buf.push_back(std::byte{static_cast<uint8_t>((u >> (8 * i)) & 0xff)});
}
inline void write_le_f32(std::vector<std::byte>& buf, float v) {
  uint32_t u = 0;
  static_assert(sizeof(float) == 4);
  std::memcpy(&u, &v, 4);
  write_le_u32(buf, u);
}

inline uint16_t read_le_u16(const std::byte* p) {
  return static_cast<uint16_t>(static_cast<uint8_t>(p[0])) |
         (static_cast<uint16_t>(static_cast<uint8_t>(p[1])) << 8);
}
inline uint32_t read_le_u32(const std::byte* p) {
  return static_cast<uint32_t>(static_cast<uint8_t>(p[0])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
}
inline int64_t read_le_i64(const std::byte* p) {
  uint64_t u = 0;
  for (int i = 0; i < 8; ++i)
    u |= static_cast<uint64_t>(static_cast<uint8_t>(p[i])) << (8 * i);
  return static_cast<int64_t>(u);
}
inline float read_le_f32(const std::byte* p) {
  const uint32_t u = read_le_u32(p);
  float f = 0.f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::vector<std::byte> pack_event_packet(const uint16_t* x,
                                         const uint16_t* y,
                                         const int64_t* t_us,
                                         const uint8_t* p01,
                                         uint32_t n,
                                         uint16_t width,
                                         uint16_t height) {
  std::vector<std::byte> buf;
  buf.reserve(16 + static_cast<size_t>(n) * (2 + 2 + 8 + 1));
  write_le_u32(buf, kEventPacketMagic);
  write_le_u16(buf, kEventPacketVersion);
  write_le_u16(buf, 0);
  write_le_u32(buf, n);
  write_le_u16(buf, width);
  write_le_u16(buf, height);
  for (uint32_t i = 0; i < n; ++i) write_le_u16(buf, x[i]);
  for (uint32_t i = 0; i < n; ++i) write_le_u16(buf, y[i]);
  for (uint32_t i = 0; i < n; ++i) write_le_i64(buf, t_us[i]);
  for (uint32_t i = 0; i < n; ++i) buf.push_back(std::byte{p01[i]});
  return buf;
}

// Decode packet into host accumulators (append).
void unpack_event_packet(const std::byte* data,
                         size_t size,
                         std::vector<uint16_t>& xs,
                         std::vector<uint16_t>& ys,
                         std::vector<int64_t>& ts,
                         std::vector<int8_t>& ps,
                         int& width_hint,
                         int& height_hint) {
  if (size < 16) die("EventPacket too short");
  const uint32_t magic = read_le_u32(data);
  if (magic != kEventPacketMagic) die("EventPacket bad magic (expected EGPK)");
  const uint16_t ver = read_le_u16(data + 4);
  if (ver != kEventPacketVersion) die("EventPacket unsupported version");
  const uint32_t n = read_le_u32(data + 8);
  const uint16_t w = read_le_u16(data + 12);
  const uint16_t h = read_le_u16(data + 14);
  if (w > 0) width_hint = w;
  if (h > 0) height_hint = h;

  const size_t need = 16 + static_cast<size_t>(n) * (2 + 2 + 8 + 1);
  if (size < need) die("EventPacket truncated");

  const std::byte* px = data + 16;
  const std::byte* py = px + static_cast<size_t>(n) * 2;
  const std::byte* pt = py + static_cast<size_t>(n) * 2;
  const std::byte* pp = pt + static_cast<size_t>(n) * 8;

  const size_t base = xs.size();
  xs.resize(base + n);
  ys.resize(base + n);
  ts.resize(base + n);
  ps.resize(base + n);
  for (uint32_t i = 0; i < n; ++i) {
    xs[base + i] = read_le_u16(px + static_cast<size_t>(i) * 2);
    ys[base + i] = read_le_u16(py + static_cast<size_t>(i) * 2);
    ts[base + i] = read_le_i64(pt + static_cast<size_t>(i) * 8);
    const uint8_t p01 = static_cast<uint8_t>(pp[i]);
    ps[base + i] = p01 ? int8_t{1} : int8_t{-1};
  }
}

std::vector<std::byte> pack_imu_sample(const ImuSample& s) {
  std::vector<std::byte> buf;
  buf.reserve(32);
  write_le_i64(buf, s.t_us);
  write_le_f32(buf, s.gyro[0]);
  write_le_f32(buf, s.gyro[1]);
  write_le_f32(buf, s.gyro[2]);
  write_le_f32(buf, s.accel[0]);
  write_le_f32(buf, s.accel[1]);
  write_le_f32(buf, s.accel[2]);
  return buf;
}

ImuSample unpack_imu_sample(const std::byte* data, size_t size) {
  if (size < 32) die("ImuSample message too short");
  ImuSample s;
  s.t_us = read_le_i64(data);
  s.gyro[0] = read_le_f32(data + 8);
  s.gyro[1] = read_le_f32(data + 12);
  s.gyro[2] = read_le_f32(data + 16);
  s.accel[0] = read_le_f32(data + 20);
  s.accel[1] = read_le_f32(data + 24);
  s.accel[2] = read_le_f32(data + 28);
  return s;
}

} // namespace

void write_events_mcap(const std::string& path,
                       int width,
                       int height,
                       const std::string& sensor,
                       const uint16_t* x,
                       const uint16_t* y,
                       const int64_t* t_us,
                       const uint8_t* p01,
                       int64_t n_events,
                       const std::vector<ImuSample>* imu,
                       int events_per_packet) {
  if (events_per_packet <= 0) events_per_packet = 65536;

  mcap::McapWriter writer;
  mcap::McapWriterOptions opts("eventgate");
  opts.compression = mcap::Compression::None;
  opts.noChunkCRC = true;
  const auto st = writer.open(path, opts);
  if (!st.ok()) die("McapWriter open failed: " + std::string(st.message));

  mcap::Schema ev_schema(kMcapSchemaEvents, "jsonschema", kEventPacketSchemaJson);
  writer.addSchema(ev_schema);
  mcap::Channel ev_ch(kMcapTopicEvents, "raw", ev_schema.id);
  writer.addChannel(ev_ch);

  mcap::Schema imu_schema(kMcapSchemaImu, "jsonschema", kImuSampleSchemaJson);
  writer.addSchema(imu_schema);
  mcap::Channel imu_ch(kMcapTopicImu, "raw", imu_schema.id);
  writer.addChannel(imu_ch);

  mcap::Metadata meta;
  meta.name = kMcapMetadataName;
  meta.metadata["width"] = std::to_string(width);
  meta.metadata["height"] = std::to_string(height);
  meta.metadata["sensor"] = sensor;
  meta.metadata["schema_version"] = std::to_string(kEventPacketVersion);
  if (auto mst = writer.write(meta); !mst.ok())
    die("write metadata failed: " + std::string(mst.message));

  uint32_t seq = 0;
  for (int64_t off = 0; off < n_events; off += events_per_packet) {
    const int64_t n64 = std::min<int64_t>(events_per_packet, n_events - off);
    const uint32_t n = static_cast<uint32_t>(n64);
    auto packet = pack_event_packet(x + off, y + off, t_us + off, p01 + off, n,
                                    static_cast<uint16_t>(width), static_cast<uint16_t>(height));
    mcap::Message msg;
    msg.channelId = ev_ch.id;
    msg.sequence = seq++;
    const int64_t t0 = (n_events > 0) ? t_us[off] : 0;
    msg.logTime = static_cast<uint64_t>(t0) * 1000ull; // us → ns
    msg.publishTime = msg.logTime;
    msg.data = packet.data();
    msg.dataSize = packet.size();
    if (auto wst = writer.write(msg); !wst.ok())
      die("write EventPacket failed: " + std::string(wst.message));
  }

  if (imu) {
    uint32_t iseq = 0;
    for (const ImuSample& s : *imu) {
      auto packet = pack_imu_sample(s);
      mcap::Message msg;
      msg.channelId = imu_ch.id;
      msg.sequence = iseq++;
      msg.logTime = static_cast<uint64_t>(s.t_us) * 1000ull;
      msg.publishTime = msg.logTime;
      msg.data = packet.data();
      msg.dataSize = packet.size();
      if (auto wst = writer.write(msg); !wst.ok())
        die("write ImuSample failed: " + std::string(wst.message));
    }
  }

  writer.close();
}

void write_synthetic_mcap(const std::string& path,
                          int width,
                          int height,
                          int64_t duration_us,
                          int64_t active_us,
                          int events_per_active_us,
                          bool with_imu) {
  if (active_us > duration_us) active_us = duration_us;
  const int64_t n = active_us * static_cast<int64_t>(events_per_active_us);
  std::vector<uint16_t> x(static_cast<size_t>(n));
  std::vector<uint16_t> y(static_cast<size_t>(n));
  std::vector<int64_t> t(static_cast<size_t>(n));
  std::vector<uint8_t> p(static_cast<size_t>(n));

  std::mt19937_64 rng(0xC0FFEE);
  std::uniform_int_distribution<int> dx(0, width - 1);
  std::uniform_int_distribution<int> dy(0, height - 1);
  std::uniform_int_distribution<int> pol(0, 1);

  for (int64_t i = 0; i < n; ++i) {
    t[static_cast<size_t>(i)] =
        static_cast<int64_t>((static_cast<double>(i) / static_cast<double>(std::max<int64_t>(n, 1))) *
                             static_cast<double>(active_us));
    x[static_cast<size_t>(i)] = static_cast<uint16_t>(dx(rng));
    y[static_cast<size_t>(i)] = static_cast<uint16_t>(dy(rng));
    p[static_cast<size_t>(i)] = static_cast<uint8_t>(pol(rng));
  }
  std::sort(t.begin(), t.end());

  std::vector<ImuSample> imu_samples;
  const std::vector<ImuSample>* imu_ptr = nullptr;
  if (with_imu) {
    const ImuCsv synth = ImuCsv::make_synthetic(duration_us, 200.f);
    imu_samples.reserve(synth.size());
    for (size_t i = 0; i < synth.size(); ++i) imu_samples.push_back(synth.at(i));
    imu_ptr = &imu_samples;
  }

  write_events_mcap(path, width, height, "IMX637_synthetic", x.data(), y.data(), t.data(), p.data(),
                    n, imu_ptr);
}

EventMcap::EventMcap(const std::string& path) {
  mcap::McapReader reader;
  const auto st = reader.open(path);
  if (!st.ok()) die("McapReader open failed: " + path + " : " + std::string(st.message));

  // Optional summary indices (helps channel discovery).
  (void)reader.readSummary(mcap::ReadSummaryMethod::AllowFallbackScan);

  // Metadata
  meta_.width = kDefaultWidth;
  meta_.height = kDefaultHeight;
  meta_.sensor = "unknown";
  for (const auto& [name, md] : reader.metadata()) {
    if (name != kMcapMetadataName && name != "eventgate") continue;
    for (const auto& kv : md.metadata) {
      if (kv.first == "width") meta_.width = std::stoi(kv.second);
      else if (kv.first == "height") meta_.height = std::stoi(kv.second);
      else if (kv.first == "sensor") meta_.sensor = kv.second;
    }
  }

  std::vector<uint16_t> xs, ys;
  std::vector<int64_t> ts;
  std::vector<int8_t> ps;

  auto on_problem = [](const mcap::Status& problem) {
    throw std::runtime_error(std::string("mcap read: ") + std::string(problem.message));
  };

  for (const auto& view : reader.readMessages(on_problem)) {
    const auto& ch = view.channel;
    if (!ch) continue;
    if (ch->topic == kMcapTopicEvents) {
      if (ch->messageEncoding != "raw")
        die(std::string("unexpected events encoding: ") + ch->messageEncoding);
      unpack_event_packet(view.message.data, view.message.dataSize, xs, ys, ts, ps, meta_.width,
                          meta_.height);
    } else if (ch->topic == kMcapTopicImu) {
      if (ch->messageEncoding != "raw")
        die(std::string("unexpected imu encoding: ") + ch->messageEncoding);
      imu_.push(unpack_imu_sample(view.message.data, view.message.dataSize));
    }
  }

  reader.close();

  n_ = static_cast<int64_t>(xs.size());
  meta_.n_events = n_;
  if (n_ > 0) {
    // Sort by timestamp if packets arrived out of order (should already be ordered).
    std::vector<size_t> order(static_cast<size_t>(n_));
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return ts[a] < ts[b]; });

    x_ = static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * static_cast<size_t>(n_)));
    y_ = static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * static_cast<size_t>(n_)));
    t_ = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * static_cast<size_t>(n_)));
    p_ = static_cast<int8_t*>(std::malloc(sizeof(int8_t) * static_cast<size_t>(n_)));
    if (!x_ || !y_ || !t_ || !p_) die("OOM loading MCAP events");

    for (int64_t i = 0; i < n_; ++i) {
      const size_t j = order[static_cast<size_t>(i)];
      x_[static_cast<size_t>(i)] = xs[j];
      y_[static_cast<size_t>(i)] = ys[j];
      t_[static_cast<size_t>(i)] = ts[j];
      p_[static_cast<size_t>(i)] = ps[j];
    }
    meta_.t_min_us = t_[0];
    meta_.t_max_us = t_[static_cast<size_t>(n_ - 1)];
  }
}

EventMcap::~EventMcap() {
  std::free(x_);
  std::free(y_);
  std::free(t_);
  std::free(p_);
}

std::pair<int64_t, int64_t> EventMcap::index_range(int64_t t0_us, int64_t t1_us) const {
  if (n_ == 0) return {0, 0};
  const int64_t* begin = t_;
  const int64_t* end = t_ + n_;
  const int64_t* i0 = std::lower_bound(begin, end, t0_us);
  const int64_t* i1 = std::lower_bound(begin, end, t1_us);
  return {static_cast<int64_t>(i0 - begin), static_cast<int64_t>(i1 - begin)};
}

EventSoA EventMcap::view(int64_t i0, int64_t i1) const {
  EventSoA v;
  if (i1 <= i0) {
    v.n = 0;
    return v;
  }
  v.x = x_ + i0;
  v.y = y_ + i0;
  v.t_us = t_ + i0;
  v.p = p_ + i0;
  v.n = static_cast<int32_t>(i1 - i0);
  return v;
}

void convert_hdf5_to_mcap(const std::string& h5_path,
                          const std::string& mcap_path,
                          const std::string& imu_csv_path) {
  EventHdf5 h5(h5_path);
  const auto& m = h5.meta();
  const int64_t n = h5.size();

  // p is ±1 in memory — convert to 0/1 for wire format
  std::vector<uint8_t> p01(static_cast<size_t>(std::max<int64_t>(n, 0)));
  const int8_t* src_p = h5.p();
  for (int64_t i = 0; i < n; ++i)
    p01[static_cast<size_t>(i)] = (src_p[static_cast<size_t>(i)] > 0) ? uint8_t{1} : uint8_t{0};

  std::vector<ImuSample> imu_samples;
  const std::vector<ImuSample>* imu_ptr = nullptr;
  if (!imu_csv_path.empty()) {
    ImuCsv csv(imu_csv_path);
    imu_samples.reserve(csv.size());
    for (size_t i = 0; i < csv.size(); ++i) imu_samples.push_back(csv.at(i));
    if (!imu_samples.empty()) imu_ptr = &imu_samples;
  }

  write_events_mcap(mcap_path, m.width, m.height, m.sensor.empty() ? "IMX637" : m.sensor, h5.x(),
                    h5.y(), h5.t(), p01.data(), n, imu_ptr);
}

} // namespace eventgate
