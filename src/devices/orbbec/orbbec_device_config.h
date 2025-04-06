#pragma once

#include "../device_config.gen.h"

namespace pc::devices {

struct OrbbecDeviceConfiguration {
  std::string id;
  bool active = true;
  int depth_mode = 0;                     // @minmax(0, 1)
  int acquisition_mode = 0;               // @minmax(0, 1)
  DeviceTransformConfiguration transform; // @optional
};

} // namespace pc::devices