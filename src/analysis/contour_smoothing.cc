#include "contour_smoothing.h"
#include "analyser_2d_config.gen.h"
#include <opencv2/core/types.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace pc::analysis::contour_smoothing {

static void chaikin_pass(const std::vector<cv::Point2f> &contour_input,
                         std::vector<cv::Point2f> &output_points,
                         const ContourSmoothingConfiguration &config) {
  const auto point_count = contour_input.size();
  if (point_count < 2) {
    output_points = contour_input;
    return;
  }

  const auto corner_cutting = std::clamp(config.corner_cutting, 1e-4f, 0.4999f);
  output_points.clear();
  output_points.reserve(point_count * 2);

  const float preserve_sharp_cos =
      config.preserve_sharp_corners
          ? std::cos(config.sharp_corner_angle * CV_PI / 180.0f)
          : 0.0f;

  for (std::size_t i = 0; i < point_count; ++i) {
    const auto &p_prev = contour_input[(i + point_count - 1) % point_count];
    const auto &p0 = contour_input[i];
    const auto &p1 = contour_input[(i + 1) % point_count];
    if (config.preserve_sharp_corners) {
      constexpr auto minimum_segment_length = 1e-6f;
      const auto v0 = p0 - p_prev;
      const auto v1 = p1 - p0;
      const float l0 = cv::norm(v0);
      const float l1 = cv::norm(v1);
      if (l0 > minimum_segment_length && l1 > minimum_segment_length) {
        const float cos_angle = (v0.dot(v1)) / (l0 * l1);
        if (cos_angle >= preserve_sharp_cos) {
          output_points.emplace_back(p0);
          continue;
        }
      }
    }
    output_points.emplace_back((1.0f - corner_cutting) * p0 +
                               corner_cutting * p1);
    output_points.emplace_back(corner_cutting * p0 +
                               (1.0f - corner_cutting) * p1);
  }
}

// Chaikin's algorithm for iterative contour smoothing
std::vector<cv::Point2f>
chaikin_smooth(const std::vector<cv::Point2f> &contour_input,
               const ContourSmoothingConfiguration &config) {
  auto iterations = std::max(0, config.iterations);
  std::vector<cv::Point2f> a = contour_input;
  std::vector<cv::Point2f> b;
  for (int i = 0; i < iterations; ++i) {
    chaikin_pass(a, b, config);
    a.swap(b);
  }
  return a;
}

// resample a 2d contour to have a specific number of output vertices
// (arc length reparametisation)
std::vector<cv::Point2f>
resample_by_arclength(const std::vector<cv::Point2f> &contour_input,
                      int resample_count) {
  std::vector<cv::Point2f> contour_output;
  const auto vertex_count = contour_input.size();
  if (vertex_count < 2 || resample_count < 2) return contour_output;

  // build cumulative arclength along the contour
  std::vector<float> cumulative_lengths;
  cumulative_lengths.reserve(vertex_count + 1);

  float total_length = 0.0f;
  cumulative_lengths.push_back(0.0f);

  for (std::size_t i = 0; i < vertex_count; ++i) {
    const auto &a = contour_input[i];
    const auto &b = contour_input[(i + 1) % vertex_count];
    total_length += cv::norm(b - a);
    cumulative_lengths.push_back(total_length);
  }

  contour_output.reserve(resample_count);

  const auto step = total_length / resample_count;

  std::size_t current_segment = 0;
  for (int k = 0; k < resample_count; k++) {
    const float target_distance = step * k;
    while (current_segment + 1 < cumulative_lengths.size() &&
           cumulative_lengths[current_segment + 1] < target_distance) {
      current_segment++;
    }
    current_segment = std::min(current_segment, vertex_count - 1);

    const float seg_start_distance = cumulative_lengths[current_segment];
    const float seg_end_distance = cumulative_lengths[current_segment + 1];
    const float segment_length = seg_end_distance - seg_start_distance;
    float t = 0.0f;
    if (segment_length > std::numeric_limits<float>::epsilon()) {
      t = (target_distance - seg_start_distance) / segment_length;
    }

    const auto &p0 = contour_input[current_segment];
    const auto &p1 = contour_input[(current_segment + 1) % vertex_count];
    contour_output.emplace_back((1.0f - t) * p0 + t * p1);
  }

  return contour_output;
}

} // namespace pc::analysis::contour_smoothing