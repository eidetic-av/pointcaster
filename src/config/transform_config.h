#pragma once

#include <pointcaster/core_types.h>

namespace pc {

struct TransformConfiguration {
  float3 position; // @minmax(-10, 10)
  float3 rotation; // @minmax(-360, 360)
  float3 scale; // @minmax(0, 2.5)
};

} // namespace pc