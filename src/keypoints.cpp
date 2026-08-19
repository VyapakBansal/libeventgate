#include "libeventgate/keypoints.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace eventgate {

cv::Mat frame_to_u8(const float* hw, int height, int width) {
  cv::Mat f32(height, width, CV_32FC1, const_cast<float*>(hw));
  return frame_to_u8(f32);
}

cv::Mat frame_to_u8(const cv::Mat& f32) {
  CV_Assert(f32.type() == CV_32FC1);
  double mn = 0, mx = 0;
  cv::minMaxLoc(f32, &mn, &mx);
  cv::Mat u8;
  if (mx > mn) {
    f32.convertTo(u8, CV_8UC1, 255.0 / (mx - mn), -mn * 255.0 / (mx - mn));
  } else {
    u8 = cv::Mat::zeros(f32.size(), CV_8UC1);
  }
  return u8;
}

int count_keypoints(const cv::Mat& gray_or_bgr, const std::string& detector, int max_features) {
  cv::Mat gray;
  if (gray_or_bgr.channels() == 1) {
    gray = gray_or_bgr;
  } else {
    cv::cvtColor(gray_or_bgr, gray, cv::COLOR_BGR2GRAY);
  }

  std::vector<cv::KeyPoint> kps;
  if (detector == "ORB" || detector == "orb") {
    auto orb = cv::ORB::create(max_features);
    orb->detect(gray, kps);
  } else if (detector == "FAST" || detector == "fast") {
    auto fast = cv::FastFeatureDetector::create();
    fast->detect(gray, kps);
    if (static_cast<int>(kps.size()) > max_features) kps.resize(static_cast<size_t>(max_features));
  } else {
    throw std::runtime_error("unknown detector: " + detector);
  }
  return static_cast<int>(kps.size());
}

KeypointCsvWriter::KeypointCsvWriter(const std::string& path) : out_(path) {
  if (!out_) throw std::runtime_error("cannot open keypoint csv: " + path);
}

void KeypointCsvWriter::write(double t_s, int count) {
  if (!header_) {
    out_ << "t_s,count\n";
    header_ = true;
  }
  out_ << t_s << ',' << count << '\n';
}

void KeypointCsvWriter::close() {
  if (out_.is_open()) out_.close();
}

ReconVideoWriter::ReconVideoWriter(const std::string& path, int width, int height, double fps) {
  // Color flag true: write_gray converts GRAY to BGR because mp4v/MJPG expect 3 channels.
  int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  ok_ = vw_.open(path, fourcc, fps, cv::Size(width, height), /*isColor=*/true);
  if (!ok_) {
    fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    const std::string avi = path.substr(0, path.find_last_of('.')) + ".avi";
    ok_ = vw_.open(avi.empty() ? path : avi, fourcc, fps, cv::Size(width, height), true);
  }
}

void ReconVideoWriter::write_gray(const cv::Mat& gray_u8) {
  if (!ok_) return;
  // Most codecs reject single-channel frames; convert here, not at the caller.
  cv::Mat bgr;
  if (gray_u8.channels() == 1) {
    cv::cvtColor(gray_u8, bgr, cv::COLOR_GRAY2BGR);
  } else {
    bgr = gray_u8;
  }
  vw_.write(bgr);
}

void ReconVideoWriter::release() {
  if (vw_.isOpened()) vw_.release();
  ok_ = false;
}

} // namespace eventgate
