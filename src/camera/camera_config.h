#pragma once

#include "../point_cloud_renderer_config.h"
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Vector3.h>
#include <array>
#include <string>
#include <zpp_bits.h>

namespace pc::camera {

using Euler = Magnum::Math::Vector3<Magnum::Math::Rad<float>>;
using Position = Magnum::Math::Vector3<float>;
using Deg_f = Magnum::Math::Deg<float>;
using Rad_f = Magnum::Math::Rad<float>;

namespace defaults {

static constexpr std::array<float, 3> rotation{15, 0, 0};
static constexpr float distance = 10;
static const std::array<float, 3> translation{
    0.0f, distance *std::sin(float(Rad_f(rotation[0]))),
    distance *std::cos(float(Rad_f(rotation[0])))};
static constexpr float fov = 45;

namespace magnum {

static constexpr Euler rotation{Deg_f{defaults::rotation[0]},
                                Deg_f{defaults::rotation[1]},
                                Deg_f{defaults::rotation[2]}};
static constexpr float distance = defaults::distance;
static const Position translation{defaults::translation[0],
                                  defaults::translation[1],
                                  defaults::translation[2]};
static constexpr Deg_f fov{defaults::fov};

} // namespace magnum

} // namespace defaults

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
  float minimum_area = 0.0f;
};

struct ContourDetectionConfiguration {
  bool enabled = false;
  bool draw = false;
  bool publish = false;
  bool simplify = false;
  float simplify_arc_scale = 0.001f;
  float simplify_min_area = 0.0001f;
  TriangulateConfiguration triangulate;
};

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

struct FrameAnalysisConfiguration {
  bool enabled;
  std::array<int, 2> resolution;
  std::array<int, 2> binary_threshold = {50, 255};
  bool greyscale_conversion = true;
  int blur_size = 1;

  CannyEdgeConfiguration canny;

  ContourDetectionConfiguration contours;
  bool contours_open;
  
  OpticalFlowConfiguration optical_flow;
  bool optical_flow_open;
};

struct CameraConfiguration {
  std::string id;
  std::string name;
  bool show_window;

  std::array<float, 3> rotation = defaults::rotation;
  std::array<float, 3> translation = defaults::translation;
  float fov = defaults::fov;
  bool transform_open;

  PointCloudRendererConfiguration rendering;
  bool rendering_open;

  FrameAnalysisConfiguration frame_analysis;
  bool analysis_open;
};

} // namespace pc::camera
