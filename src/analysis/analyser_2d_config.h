#pragma once
#include <array>
#include <serdepp/serde.hpp>
#include "../structs.h"

namespace pc::analysis {

using pc::types::Int2;
using pc::types::Float2;

struct CannyEdgeConfiguration {
  bool enabled = false;
  int min_threshold = 100;
  int max_threshold = 255;
  int aperture_size = 3;
};

struct SimplifyConfiguration {
  bool enabled = false;
};

struct TriangulateConfiguration {
  bool enabled = false;
  bool draw = false;
  bool publish = false;
  float minimum_area = 0.0f; // @minmax(0, 10)
};

struct ContourSmoothingConfiguration {
  bool enabled = false;
  int iterations = 3; // @minmax(1, 6)
  int resample_count = 1000; // @minmax(10, 5000)
  float corner_cutting = 0.25; // @minmax(0.05, 0.45)
  bool preserve_sharp_corners = false;
  float sharp_corner_angle = 45; // @minmax(20, 90)
};

struct ContourDetectionConfiguration {
  bool enabled = false;
  bool draw = false;
  bool label = false;
  bool publish = false;
  bool publish_centroids = false;
  bool simplify = false;
  float simplify_arc_scale = 0.001f; // @minmax(0.00001f, 10.0f)
  float simplify_min_area = 0.0001f; // @minmax(0.00001f, 10.0f)
  ContourSmoothingConfiguration smoothing; // @optional
  TriangulateConfiguration triangulate;
};

struct OpticalFlowConfiguration {
  bool enabled = false;
  bool draw = false;
  bool publish = false;
  int feature_point_count = 250;
  float feature_point_distance = 10.0f;
  float cuda_feature_detector_quality_cutoff = 0.01f;
  float magnitude_scale = 1.0f;
  float magnitude_exponent = 1.0f;
  float minimum_distance = 0.0f;
  float maximum_distance = 0.5f;
};

struct OutputConfiguration {
  bool unfolded = false;
  Float2 scale{1.0f, 1.0f};
  Float2 offset{0.0f, 0.0f};
};

struct Analyser2DConfiguration {
  bool unfolded = false;
  bool enabled = false;
  bool use_cuda = false;
  Int2 resolution{480, 270}; // @minmax(2, 4096)
  Int2 binary_threshold{50, 255}; // @minmax(0, 255)
  int blur_size = 1;

  CannyEdgeConfiguration canny;
  ContourDetectionConfiguration contours;
  OpticalFlowConfiguration optical_flow;
  OutputConfiguration output;
};

} // namespace pc::analysis
