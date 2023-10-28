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

  DERIVE_SERDE(CannyEdgeConfiguration,
               (&Self::enabled, "enabled")
               (&Self::min_threshold, "min_threshold")
               (&Self::max_threshold, "max_threshold")
               (&Self::aperture_size, "aperture_size"))
};

struct SimplifyConfiguration {
  bool enabled = false;

  DERIVE_SERDE(SimplifyConfiguration, (&Self::enabled, "enabled"))
};

struct TriangulateConfiguration {
  bool enabled = false;
  bool draw = false;
  bool publish = false;
  float minimum_area = 0.0f;

  DERIVE_SERDE(TriangulateConfiguration,
               (&Self::enabled, "enabled")
               (&Self::draw, "draw")
               (&Self::publish, "publish")
               (&Self::minimum_area, "minimum_area"))
};

struct ContourDetectionConfiguration {
  bool unfolded = false;
  bool enabled = false;
  bool draw = false;
  bool label = false;
  bool publish = false;
  bool publish_centroids = false;
  bool simplify = false;
  float simplify_arc_scale = 0.001f;
  float simplify_min_area = 0.0001f;
  TriangulateConfiguration triangulate;

  DERIVE_SERDE(ContourDetectionConfiguration,
               (&Self::unfolded, "unfolded")
               (&Self::enabled, "enabled")
               (&Self::draw, "draw")
               (&Self::label, "label")
               (&Self::publish, "publish")
               (&Self::publish_centroids, "publish_centroids")
               (&Self::simplify, "simplify")
               (&Self::simplify_arc_scale, "simplify_arc_scale")
               (&Self::simplify_min_area, "simplify_min_area")
               (&Self::triangulate, "triangulate"))
};

struct OpticalFlowConfiguration {
  bool unfolded = false;
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

  DERIVE_SERDE(OpticalFlowConfiguration,
               (&Self::unfolded, "unfolded")
               (&Self::enabled, "enabled")
               (&Self::draw, "draw")
               (&Self::publish, "publish")
               (&Self::feature_point_count, "feature_point_count")
               (&Self::feature_point_distance, "feature_point_distance")
               (&Self::cuda_feature_detector_quality_cutoff, "cuda_feature_detector_quality_cutoff")
               (&Self::magnitude_scale, "magnitude_scale")
               (&Self::magnitude_exponent, "magnitude_exponent")
               (&Self::minimum_distance, "minimum_distance")
               (&Self::maximum_distance, "maximum_distance"))
};

struct OutputConfiguration {
  bool unfolded = false;
  Float2 scale = {1.0f, 1.0f};
  Float2 offset = {0.0f, 0.0f};

  DERIVE_SERDE(OutputConfiguration,
               (&Self::unfolded, "unfolded")
               (&Self::scale, "scale")
               (&Self::offset, "offset"))
};

struct Analyser2DConfiguration {
  bool unfolded = false;
  bool enabled = false;
  bool use_cuda = false;
  Int2 resolution = {480, 270};
  Int2 binary_threshold = {50, 255};
  int blur_size = 1;

  CannyEdgeConfiguration canny;
  ContourDetectionConfiguration contours;
  OpticalFlowConfiguration optical_flow;
  OutputConfiguration output;

  DERIVE_SERDE(Analyser2DConfiguration,
               (&Self::unfolded, "unfolded")
               (&Self::enabled, "enabled")
               (&Self::use_cuda, "use_cuda")
               (&Self::resolution, "resolution")
               (&Self::binary_threshold, "binary_threshold")
               (&Self::blur_size, "blur_size")
               (&Self::canny, "canny")
               (&Self::contours, "contours")
               (&Self::optical_flow, "optical_flow")
               (&Self::output, "output"))
};

} // namespace pc::analysis
