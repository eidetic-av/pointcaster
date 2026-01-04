#pragma once
#include <string>

namespace pc::devices {

struct OrbbecDevice;

struct OrbbecDeviceConfiguration {
  std::string id;           // @hidden
  bool active = true;       // @hidden
  std::string ip;           // @disabled
  int depth_mode = 0;       // @minmax(0, 1)
  int acquisition_mode = 0; // @minmax(0, 1)

  using DeviceType = OrbbecDevice;
  static constexpr auto PublishPath = "ob";
};

} // namespace pc::devices