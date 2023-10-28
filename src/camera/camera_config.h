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

using pc::types::Float3;
using pc::types::Int2;
using pc::types::MinMax;

using Euler = Magnum::Math::Vector3<Magnum::Math::Rad<float>>;
using Position = Magnum::Math::Vector3<float>;
using Deg_f = Magnum::Math::Deg<float>;
using Rad_f = Magnum::Math::Rad<float>;

namespace defaults {

static constexpr float distance = 3.5f;
static constexpr Float2 orbit = {0.0f, 15.0f};
static constexpr float roll = 0.0f;
static constexpr Float3 translation{0.0f, 0.0f, 0.0f};
static constexpr float fov = 45;

static constexpr Int2 rendering_resolution{3840, 2160};
static constexpr Int2 analysis_resolution{480, 270};

static constexpr MinMax<float> perspective_clipping = {0.0001f, 20.0f};
static constexpr MinMax<float> orthographic_clipping = {-15, 15};

} // namespace defaults

struct TransformConfiguration {
  bool unfolded = true;
  bool show_anchor = false;
  float distance = defaults::distance;
  Float2 orbit = defaults::orbit;
  float roll = defaults::roll;
  Float3 translation = defaults::translation;

  DERIVE_SERDE(TransformConfiguration,
	       (&Self::unfolded, "unfolded")
	       (&Self::show_anchor, "show_anchor")
	       (&Self::distance, "distance")
	       (&Self::orbit, "orbit")
	       (&Self::roll, "roll")
	       (&Self::translation, "translation"))

  using MemberTypes = pc::reflect::type_list<bool, bool, float, Float2, float, Float3>;
  static const std::size_t MemberCount = 6;
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
