#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <string>
#include <vector>

namespace eventgate {

struct KeypointSample {
  double t_s = 0.0;
  int count  = 0;
};

// ORB/FAST on reconstructed intensity frame. Count only (no descriptors needed for pitch plot).
int count_keypoints(const cv::Mat& gray_or_bgr, const std::string& detector, int max_features);

// Convert FireNet float frame [0,1] or arbitrary range → 8-bit gray for ORB/video.
cv::Mat frame_to_u8(const float* hw, int height, int width); // row-major HxW
cv::Mat frame_to_u8(const cv::Mat& f32);

// Append CSV: t_s,count  (header written once)
class KeypointCsvWriter {
public:
  explicit KeypointCsvWriter(const std::string& path);
  void write(double t_s, int count);
  void close();

private:
  std::ofstream out_;
  bool header_ = false;
};

// Minimal MP4 (or AVI fallback) writer for demo packaging.
class DemoVideoWriter {
public:
  DemoVideoWriter(const std::string& path, int width, int height, double fps);
  void write_gray(const cv::Mat& gray_u8);
  void release();
  [[nodiscard]] bool ok() const noexcept { return ok_; }

private:
  cv::VideoWriter vw_;
  bool ok_ = false;
};

} // namespace eventgate
