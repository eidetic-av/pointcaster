#pragma once

#include "../structs.h"
#include <nlohmann/json.hpp>

namespace pc::sensors {

using pc::types::float3;
using pc::types::minMax;
using pc::types::short3;

struct BodyTrackingConfiguration {
  bool enabled;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BodyTrackingConfiguration, enabled);

struct DeviceConfiguration {
  bool flip_x, flip_y, flip_z;
  minMax<short> crop_x{-10000, 10000};
  minMax<short> crop_y{-10000, 10000};
  minMax<short> crop_z{-10000, 10000};
  short3 offset;
  float3 rotation_deg{0, 0, 0};
  float scale;
  int sample = 1;

  BodyTrackingConfiguration body;
  bool body_open;

};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DeviceConfiguration, flip_x, flip_y, flip_z,
				   crop_x, crop_y, crop_z, offset, rotation_deg,
				   scale, sample, body, body_open);
} // namespace pc::sensors
