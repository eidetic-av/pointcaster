#pragma once

#include "../analysis/analyser_2d_config.h"
#include "../point_cloud_renderer_config.h"
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Vector3.h>
#include <array>
#include <nlohmann/json.hpp>
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

static constexpr std::array<int, 2> rendering_resolution{3840, 2160};
static constexpr std::array<int, 2> analysis_resolution{480, 270};

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

  pc::analysis::Analyser2DConfiguration frame_analysis;
  bool analysis_open;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CameraConfiguration, id, name, show_window,
                                   rotation, translation, fov, transform_open,
                                   rendering, rendering_open, frame_analysis,
                                   analysis_open);

} // namespace pc::camera
