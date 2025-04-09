#pragma once

#include "../analysis/analyser_2d_config.gen.h"
#include "../point_cloud_renderer_config.gen.h"
#include "../serialization.h"
#include "../structs.h"
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Vector3.h>
#include <array>
#include <string>
#include <zpp_bits.h>

namespace pc::camera {

using pc::types::Int;
using pc::types::Float;
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
static constexpr Float roll{0.0f};
static constexpr Float3 translation{0.0f, 0.0f, 0.0f};
static constexpr float fov = 45;

static constexpr Int2 rendering_resolution{3840, 2160};
static constexpr Int2 analysis_resolution{480, 270};

static constexpr MinMax<float> perspective_clipping = {0.0001f, 20.0f};
static constexpr MinMax<float> orthographic_clipping = {-15, 15};

} // namespace defaults

struct TransformConfiguration {
  bool show_anchor;
  float distance = 3.5;
  Float2 orbit{0.0f, 15.0f}; // @minmax(-360, 360)
  float roll; // @minmax(-180, 180)
  Float3 translation;
};

struct CameraConfiguration {
  std::string id;
  std::string name;
  bool show_window{false};
  float fov = defaults::fov;
  int scroll_precision{1};
  TransformConfiguration transform; // @optional
  PointCloudRendererConfiguration rendering; // @optional
  analysis::Analyser2DConfiguration analysis; // @optional
};

} // namespace pc::camera
