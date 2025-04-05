#pragma once

#include "../device_config.gen.h"

namespace pc::devices {

struct AzureKinectDriverConfiguration {
  int exposure = 10000;
  int brightness = 128;
  int contrast = 5;
  int saturation = 32;
  int gain = 128;
};

struct AzureKinectConfiguration {
  bool unfolded = true;
  std::string id;
  bool active = true;
  DeviceTransformConfiguration transform; // @optional
  AzureKinectDriverConfiguration driver;  // @optional
  BodyTrackingConfiguration body; // @optional
  AutoTiltConfiguration auto_tilt; // @optional
};

} // namespace pc::devices