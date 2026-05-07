#pragma once

#include <config/color_transform_config.h>
#include <config/file_config.h>
#include <config/transform_config.h>
#include <rfl/Literal.hpp>
#include <string>


namespace pc::devices {

class PlyDevice;

struct PlyDeviceConfiguration {
  std::string id;     // @hidden
  bool active = true; // @hidden
  bool render = true; // @hidden;

  FileConfiguration file;
  TransformConfiguration transform;
  ColorTransformConfiguration color;

  using DeviceType = PlyDevice;
  using Tag = rfl::Literal<"ply">;
  static constexpr auto PublishPath = "ply";
  static constexpr auto PluginName = "PlyDevice";
};

} // namespace pc::devices