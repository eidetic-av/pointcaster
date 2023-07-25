#pragma once

#include "../structs.h"

namespace pc::sensors {

using pc::types::float3;
using pc::types::minMax;
using pc::types::short3;

struct DeviceConfiguration {
  bool flip_x, flip_y, flip_z;
  minMax<short> crop_x{-10000, 10000};
  minMax<short> crop_y{-10000, 10000};
  minMax<short> crop_z{-10000, 10000};
  short3 offset;
  float3 rotation_deg{0, 0, 0};
  float scale;
  int sample = 1;
};

}
