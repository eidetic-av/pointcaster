#pragma once

#include <camera/camera_config.h>
#include <string>

namespace pc {

struct SessionConfiguration {
  std::string id;
  std::string display_name;
  CameraConfiguration camera;
};

} // namespace pc