#pragma once
#include <rfl/Literal.hpp>
#include <string>

namespace pc::devices {

class OrbbecDevice;

struct OrbbecDeviceConfiguration {
  std::string id;            // @hidden
  bool active = true;        // @hidden
  std::string ip;            // @disabled

  enum class DepthMode { Narrow, Wide };
  DepthMode depth_mode = DepthMode::Narrow;

  int acquisition_mode = 0;  // @minmax(0, 1)
  float test_decimal = 30.0; // @minmax(-180, 180)

  using DeviceType = OrbbecDevice;
  using Tag = rfl::Literal<"orbbec">;
  static constexpr auto PublishPath = "ob";
  static constexpr auto PluginName = "OrbbecDevice";
};

} // namespace pc::devices