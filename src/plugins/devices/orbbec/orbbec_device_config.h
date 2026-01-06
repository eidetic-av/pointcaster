#pragma once
#include <string>
#include <rfl/Literal.hpp>

namespace pc::devices {

class OrbbecDevice;

struct OrbbecDeviceConfiguration {
  std::string id;           // @hidden
  bool active = true;       // @hidden
  std::string ip;           // @disabled
  int depth_mode = 0;       // @minmax(0, 1)
  int acquisition_mode = 0; // @minmax(0, 1)

  using DeviceType = OrbbecDevice;
  using Tag = rfl::Literal<"orbbec">;
  static constexpr auto PublishPath = "ob";
  static constexpr auto PluginName = "OrbbecDevice";
};

} // namespace pc::devices