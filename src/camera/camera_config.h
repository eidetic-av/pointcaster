#pragma once

#include <pointcaster/core_types.h>
#include <string>

namespace pc {

struct CameraConfiguration {
  std::string id;

  bool locked = false;
  bool orthographic = false;
  bool show_grid = true;

  pc::float3 position;
};

} // namespace pc