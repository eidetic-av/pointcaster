#pragma once

#include <array>
#include <nlohmann/json.hpp>

namespace pc::analysis {

struct CannyEdgeConfiguration {
  bool enabled = false;
  int min_threshold = 100;
  int max_threshold = 255;
  int aperture_size = 3;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CannyEdgeConfiguration, enabled,
                                   min_threshold, max_threshold, aperture_size);

struct SimplifyConfiguration {
  bool enabled = false;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SimplifyConfiguration, enabled);

struct TriangulateConfiguration {
  bool enabled = false;
  bool draw = false;
  bool publish = false;
  float minimum_area = 0.0f;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TriangulateConfiguration, enabled, draw,
                                   publish, minimum_area);

struct ContourDetectionConfiguration {
  bool enabled = false;
  bool draw = false;
  bool publish = false;
  bool simplify = false;
  float simplify_arc_scale = 0.001f;
  float simplify_min_area = 0.0001f;
  TriangulateConfiguration triangulate;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ContourDetectionConfiguration, enabled, draw,
                                   publish, simplify, simplify_arc_scale,
                                   simplify_min_area, triangulate);

struct OpticalFlowConfiguration {
  bool enabled = false;
  bool draw = false;
  bool publish = false;
  int feature_point_count = 250;
  float feature_point_distance = 10.0f;
  float magnitude_scale = 1.0f;
  float magnitude_exponent = 1.0f;
  float minimum_distance = 0.0f;
  float maximum_distance = 0.5f;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(OpticalFlowConfiguration, enabled, draw,
                                   publish, feature_point_count,
                                   feature_point_distance, magnitude_scale,
                                   magnitude_exponent, minimum_distance,
                                   maximum_distance);

struct Analyser2DConfiguration {
  bool enabled;
  std::array<int, 2> resolution = {480, 270};
  std::array<int, 2> binary_threshold = {50, 255};
  bool greyscale_conversion = true;
  int blur_size = 1;

  CannyEdgeConfiguration canny;

  ContourDetectionConfiguration contours;
  bool contours_open;

  OpticalFlowConfiguration optical_flow;
  bool optical_flow_open;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Analyser2DConfiguration, enabled,
                                   resolution, binary_threshold,
                                   greyscale_conversion, blur_size, canny,
                                   contours, contours_open, optical_flow,
                                   optical_flow_open);

} // namespace pc::analysis
