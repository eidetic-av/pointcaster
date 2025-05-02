#pragma once

#include "../device_config.gen.h"

namespace pc::devices {

struct K4ADevice;

struct AzureKinectDriverConfiguration {
  int exposure = 10000;
  int brightness = 128;
  int contrast = 5;
  int saturation = 32;
  int gain = 128;
};

struct AzureKinectConfiguration {
  using DeviceType = K4ADevice;
  std::string id; // @hidden
  std::string serial_number; // @disabled
  bool active = true; // @hidden
  DeviceTransformConfiguration transform; // @optional
  AzureKinectDriverConfiguration driver;  // @optional
  BodyTrackingConfiguration body; // @optional
  AutoTiltConfiguration auto_tilt; // @optional
};

} // namespace pc::devices