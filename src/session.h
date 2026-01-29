#pragma once

#include <camera/camera_config.h>
#include <string>

namespace pc {

struct SessionConfiguration {
  std::string id;
  CameraConfiguration camera;
};

} // namespace pc