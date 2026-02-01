#pragma once

#include <config/transform_config.h>
#include <pointcaster/point_cloud.h>
#include <rfl/Literal.hpp>
#include <string>

namespace pc::devices {

class OrbbecDevice;

struct OrbbecDeviceConfiguration {
  std::string id;     // @hidden
  bool active = true; // @hidden
  std::string ip;     // @disabled

  enum class DepthMode { Narrow, Wide };
  DepthMode depth_mode = DepthMode::Narrow;

  enum class AcquisitionMode { XYZRGB, XYZ };
  AcquisitionMode acquisition_mode = AcquisitionMode::XYZRGB;

  int decimation = 1; // @minmax(1, 8)

  TransformConfiguration transform;

  using DeviceType = OrbbecDevice;
  using Tag = rfl::Literal<"orbbec">;
  static constexpr auto PublishPath = "ob";
  static constexpr auto PluginName = "OrbbecDevice";
};

} // namespace pc::devices