#include "libeventgate/hdf5_events.hpp"

#include <hdf5.h>
#include <hdf5_hl.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace eventgate {
namespace {

[[noreturn]] void die(const std::string& m) { throw std::runtime_error(m); }

} // namespace

EventHdf5::EventHdf5(const std::string& path) {
  const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) die("H5Fopen failed: " + path);

  auto read_attr_int = [&](const char* name, int fallback) -> int {
    if (H5Aexists(file, name) <= 0) return fallback;
    const hid_t a = H5Aopen(file, name, H5P_DEFAULT);
    int v = fallback;
    H5Aread(a, H5T_NATIVE_INT, &v);
    H5Aclose(a);
    return v;
  };
  meta_.width  = read_attr_int("width", kDefaultWidth);
  meta_.height = read_attr_int("height", kDefaultHeight);
  if (H5Aexists(file, "sensor") > 0) {
    const hid_t a = H5Aopen(file, "sensor", H5P_DEFAULT);
    const hid_t at = H5Aget_type(a);
    if (H5Tis_variable_str(at)) {
      char* s = nullptr;
      H5Aread(a, at, &s);
      if (s) {
        meta_.sensor = s;
        H5free_memory(s);
      }
    }
    H5Tclose(at);
    H5Aclose(a);
  }

  const hid_t g = H5Gopen2(file, "events", H5P_DEFAULT);
  if (g < 0) {
    H5Fclose(file);
    die("missing /events group: " + path);
  }

  auto open_ds = [&](const char* name) -> hid_t {
    const hid_t d = H5Dopen2(g, name, H5P_DEFAULT);
    if (d < 0) die(std::string("missing dataset events/") + name);
    return d;
  };

  const hid_t dx = open_ds("x");
  const hid_t dy = open_ds("y");
  const hid_t dt = open_ds("t_us");
  const hid_t dp = open_ds("p");

  const hid_t space = H5Dget_space(dt);
  hsize_t dims[1] = {0};
  H5Sget_simple_extent_dims(space, dims, nullptr);
  H5Sclose(space);
  n_ = static_cast<int64_t>(dims[0]);
  meta_.n_events = n_;

  if (n_ > 0) {
    // Single contiguous allocations (better than vector for CUDA registration later if needed).
    x_ = static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * static_cast<size_t>(n_)));
    y_ = static_cast<uint16_t*>(std::malloc(sizeof(uint16_t) * static_cast<size_t>(n_)));
    t_ = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * static_cast<size_t>(n_)));
    p_ = static_cast<int8_t*>(std::malloc(sizeof(int8_t) * static_cast<size_t>(n_)));
    if (!x_ || !y_ || !t_ || !p_) die("OOM loading events");

    // Read to temp then convert dtypes if needed.
    {
      std::vector<uint16_t> tmp(static_cast<size_t>(n_));
      if (H5Dread(dx, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data()) < 0)
        die("read x");
      std::memcpy(x_, tmp.data(), tmp.size() * sizeof(uint16_t));
    }
    {
      std::vector<uint16_t> tmp(static_cast<size_t>(n_));
      if (H5Dread(dy, H5T_NATIVE_UINT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data()) < 0)
        die("read y");
      std::memcpy(y_, tmp.data(), tmp.size() * sizeof(uint16_t));
    }
    if (H5Dread(dt, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, t_) < 0) die("read t_us");
    {
      // p may be u8 0/1
      std::vector<uint8_t> tmp(static_cast<size_t>(n_));
      if (H5Dread(dp, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data()) < 0)
        die("read p");
      for (int64_t i = 0; i < n_; ++i) {
        p_[i] = tmp[static_cast<size_t>(i)] ? int8_t{1} : int8_t{-1};
      }
    }
    meta_.t_min_us = t_[0];
    meta_.t_max_us = t_[n_ - 1];
  }

  H5Dclose(dx);
  H5Dclose(dy);
  H5Dclose(dt);
  H5Dclose(dp);
  H5Gclose(g);
  H5Fclose(file);
}

EventHdf5::~EventHdf5() {
  std::free(x_);
  std::free(y_);
  std::free(t_);
  std::free(p_);
}

std::pair<int64_t, int64_t> EventHdf5::index_range(int64_t t0_us, int64_t t1_us) const {
  if (n_ == 0) return {0, 0};
  // lower_bound for t0, lower_bound for t1 on sorted t_
  const int64_t* begin = t_;
  const int64_t* end = t_ + n_;
  const int64_t* i0 = std::lower_bound(begin, end, t0_us);
  const int64_t* i1 = std::lower_bound(begin, end, t1_us);
  return {static_cast<int64_t>(i0 - begin), static_cast<int64_t>(i1 - begin)};
}

EventSoA EventHdf5::view(int64_t i0, int64_t i1) const {
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

void write_synthetic_events_hdf5(const std::string& path,
                                 int width,
                                 int height,
                                 int64_t duration_us,
                                 int64_t active_us,
                                 int events_per_active_us) {
  if (active_us > duration_us) active_us = duration_us;
  const int64_t n = active_us * static_cast<int64_t>(events_per_active_us);
  std::vector<uint16_t> x(static_cast<size_t>(n));
  std::vector<uint16_t> y(static_cast<size_t>(n));
  std::vector<int64_t>  t(static_cast<size_t>(n));
  std::vector<uint8_t>  p(static_cast<size_t>(n));

  std::mt19937_64 rng(0xC0FFEE);
  std::uniform_int_distribution<int> dx(0, width - 1);
  std::uniform_int_distribution<int> dy(0, height - 1);
  std::uniform_int_distribution<int> pol(0, 1);

  for (int64_t i = 0; i < n; ++i) {
    // uniform in active interval, then sort
    t[static_cast<size_t>(i)] =
        static_cast<int64_t>((static_cast<double>(i) / static_cast<double>(std::max<int64_t>(n, 1))) *
                             static_cast<double>(active_us));
    x[static_cast<size_t>(i)] = static_cast<uint16_t>(dx(rng));
    y[static_cast<size_t>(i)] = static_cast<uint16_t>(dy(rng));
    p[static_cast<size_t>(i)] = static_cast<uint8_t>(pol(rng));
  }
  std::sort(t.begin(), t.end());

  const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) die("H5Fcreate failed: " + path);

  H5LTset_attribute_int(file, ".", "width", &width, 1);
  H5LTset_attribute_int(file, ".", "height", &height, 1);
  const char* sensor = "IMX637_synthetic";
  H5LTset_attribute_string(file, ".", "sensor", sensor);

  const hid_t g = H5Gcreate2(file, "events", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  const hsize_t dims[1] = {static_cast<hsize_t>(n)};

  auto write_ds = [&](const char* name, hid_t type, const void* data) {
    const hid_t space = H5Screate_simple(1, dims, nullptr);
    const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    // chunk + shuffle + gzip for interchange files (not the hot path)
    const hsize_t chunk[1] = {static_cast<hsize_t>(std::min<int64_t>(n > 0 ? n : 1, 1 << 16))};
    H5Pset_chunk(dcpl, 1, chunk);
    H5Pset_shuffle(dcpl);
    H5Pset_deflate(dcpl, 4);
    const hid_t ds = H5Dcreate2(g, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds);
    H5Pclose(dcpl);
    H5Sclose(space);
  };

  write_ds("x", H5T_NATIVE_UINT16, x.data());
  write_ds("y", H5T_NATIVE_UINT16, y.data());
  write_ds("t_us", H5T_NATIVE_INT64, t.data());
  write_ds("p", H5T_NATIVE_UINT8, p.data());

  H5Gclose(g);
  H5Fclose(file);
  (void)duration_us; // unused: empty tail is implicit (no events past active_us)
}

} // namespace eventgate
