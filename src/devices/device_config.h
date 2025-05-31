#pragma once

#include "../serialization.h"
#include "../structs.h"

#include <k4a/k4atypes.h>
#include <variant>

namespace pc::devices {

using pc::types::Float3;
using pc::types::Float4;
using pc::types::MinMax;
using pc::types::Short3;
using MinMaxShort = pc::types::MinMax<short>;

struct BodyTrackingConfiguration {
  bool unfolded = false;
  bool enabled = false;
};

struct AutoTiltConfiguration {
  bool enabled = false;
  float lerp_factor = 0.025f;
  float threshold = 1.0f; // degrees
};

struct ChromaKeyConfiguration {
  bool enabled = false;
  float target_hue_degrees = 120; // @minmax(0, 360)
  float hue_width_degrees = 120; // @minmax(0, 360)
  float minimum_saturation = 0.135; // @minmax(0, 1)
  float minimum_value = 0.135; // @minmax(0, 1)
  bool invert_mask = false;
};

struct ColorConfiguration {
  float uniform_gain = 1.0; // @minmax(0, 3)
  ChromaKeyConfiguration chroma_key; // @optional
};

struct DeviceTransformConfiguration {
  bool unfolded = true;
  Float3 translate{0, 0, 0}; // @minmax(-10000, 10000)
  Float4 rotation{0, 0, 0, 1}; // @hidden
  Float3 rotation_deg{0, 0, 0}; // @minmax(-360, 360)
  Float3 offset{0, 0, 0}; // @minmax(-10000, 10000)
  float scale = 1; // @minmax(0, 10)
  int sample = 1;  // @minmax(1, 1000)
  bool flip_x = false; 
  bool flip_y = false;
  bool flip_z = false;
  MinMaxShort crop_x{-10000, 10000}; // @minmax(-10000, 10000)
  MinMaxShort crop_y{-10000, 10000}; // @minmax(-10000, 10000)
  MinMaxShort crop_z{-10000, 10000}; // @minmax(-10000, 10000)
  MinMaxShort bound_x{-10000, 10000}; // @minmax(-10000, 10000)
  MinMaxShort bound_y{-10000, 10000}; // @minmax(-10000, 10000)
  MinMaxShort bound_z{-10000, 10000}; // @minmax(-10000, 10000)
};

} // namespace pc::devices
