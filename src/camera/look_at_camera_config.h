#pragma once

#include <pointcaster/core_types.h>
#include <rfl/DefaultVal.hpp>
#include <string>

namespace pc {

struct LookAtCameraConfiguration {
  std::string id; // @hidden

  bool locked = false; // @hidden

  bool orthographic = false;

  rfl::DefaultVal<float3> position = float3(0, 0, -2.5); // @minmax(-10, 10)
  rfl::DefaultVal<float3> look_at_position =
      float3(0, 0, 0); // @minmax(-10, 10)

  rfl::DefaultVal<float> vertical_fov = 60.0f; // @minmax(5, 355)

  rfl::DefaultVal<int> resolution_x = 400; // @minmax(64, 4096)
  rfl::DefaultVal<int> resolution_y = 300;  // @minmax(64, 4096)

  rfl::DefaultVal<int> color_fill_passes = 0; // @minmax(0, 20)
};

} // namespace pc