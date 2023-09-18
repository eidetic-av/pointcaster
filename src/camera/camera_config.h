#pragma once

#include "../analysis/analyser_2d_config.h"
#include "../point_cloud_renderer_config.h"
#include "../serialization.h"
#include "../structs.h"
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Vector3.h>
#include <array>
#include <string>
#include <zpp_bits.h>

namespace pc::camera {

using pc::types::float3;
using pc::types::int2;
using pc::types::minMax;

using Euler = Magnum::Math::Vector3<Magnum::Math::Rad<float>>;
using Position = Magnum::Math::Vector3<float>;
using Deg_f = Magnum::Math::Deg<float>;
using Rad_f = Magnum::Math::Rad<float>;

namespace defaults {

static constexpr float3 rotation{15, 0, 0};

static constexpr float distance = 10;
static const float3 translation{0.0f,
				distance *std::sin(float(Rad_f(rotation.x))),
				distance *std::cos(float(Rad_f(rotation.x)))};
static constexpr float fov = 45;

static constexpr int2 rendering_resolution{3840, 2160};
static constexpr int2 analysis_resolution{480, 270};

static constexpr minMax<float> clipping = {0.001f, 200};

namespace magnum {

static constexpr Euler rotation{Deg_f{defaults::rotation.x},
                                Deg_f{defaults::rotation.y},
                                Deg_f{defaults::rotation.z}};
static constexpr float distance = defaults::distance;
static const Position translation{defaults::translation[0],
                                  defaults::translation[1],
                                  defaults::translation[2]};
static constexpr Deg_f fov{defaults::fov};

} // namespace magnum

} // namespace defaults

struct TransformConfiguration {
  bool unfolded;
  float3 rotation = defaults::rotation;
  float3 translation = defaults::translation;

  DERIVE_SERDE(TransformConfiguration,
	       (&Self::unfolded, "unfolded")
	       (&Self::rotation, "rotation")
	       (&Self::translation, "translation"))

  using MemberTypes = pc::reflect::type_list<bool, float3, float3>;
  static const std::size_t MemberCount = 3;
};

struct CameraConfiguration {
  std::string id;
  std::string name;
  bool show_window;
  float fov = defaults::fov;
  int scroll_precision = 1;

  TransformConfiguration transform;

  PointCloudRendererConfiguration rendering;
  analysis::Analyser2DConfiguration analysis;

  DERIVE_SERDE(CameraConfiguration,
	       (&Self::id, "id")
	       (&Self::name, "name")
	       (&Self::show_window, "show_window")
	       (&Self::fov, "fov")
	       (&Self::scroll_precision, "scroll_precision")
	       (&Self::transform, "transform")
	       (&Self::rendering, "rendering")
	       (&Self::analysis, "analysis"))

  using MemberTypes = pc::reflect::type_list<
      std::string, std::string, bool, float, int, TransformConfiguration,
      PointCloudRendererConfiguration, pc::analysis::Analyser2DConfiguration>;

  static const std::size_t MemberCount = 8;
};

} // namespace pc::camera
