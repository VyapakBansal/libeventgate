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

// Feature count only; no descriptors. Used as a reconstruction-stability metric.
int count_keypoints(const cv::Mat& gray_or_bgr, const std::string& detector, int max_features);

// Per-frame min-max to 8-bit. FireNet output is not a calibrated intensity.
cv::Mat frame_to_u8(const float* hw, int height, int width); // row-major HxW
cv::Mat frame_to_u8(const cv::Mat& f32);

class KeypointCsvWriter {
public:
  explicit KeypointCsvWriter(const std::string& path);
  void write(double t_s, int count);
  void close();

private:
  std::ofstream out_;
  bool header_ = false;
};

class ReconVideoWriter {
public:
  ReconVideoWriter(const std::string& path, int width, int height, double fps);
  void write_gray(const cv::Mat& gray_u8);
  void release();
  [[nodiscard]] bool ok() const noexcept { return ok_; }

private:
  cv::VideoWriter vw_;
  bool ok_ = false;
};

} // namespace eventgate
