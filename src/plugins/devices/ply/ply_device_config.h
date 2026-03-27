#pragma once

#include <config/transform_config.h>
#include <rfl/Literal.hpp>
#include <string>

namespace pc::devices {

class PlyDevice;

struct PlyDeviceConfiguration {
  std::string id;     // @hidden
  bool active = true; // @hidden
  std::string file_path = ""; // @file_opener

  TransformConfiguration transform;

  using DeviceType = PlyDevice;
  using Tag = rfl::Literal<"ply">;
  static constexpr auto PublishPath = "ply";
  static constexpr auto PluginName = "PlyDevice";
};

} // namespace pc::devices