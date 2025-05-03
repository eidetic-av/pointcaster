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
  std::string id; // @hidden
  std::string serial_number; // @disabled
  bool active = true; // @hidden
  DeviceTransformConfiguration transform; // @optional
  AzureKinectDriverConfiguration driver;  // @optional
  BodyTrackingConfiguration body; // @optional
  AutoTiltConfiguration auto_tilt; // @optional

  using DeviceType = K4ADevice;
  static constexpr auto PublishPath = "k4a";
};

} // namespace pc::devices