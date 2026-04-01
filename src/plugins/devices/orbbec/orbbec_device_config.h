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

  bool force_ip; // @button

  enum class DepthMode { Narrow, Wide };
  DepthMode depth_mode = DepthMode::Narrow;

  enum class AcquisitionMode { XYZRGB, XYZ };
  AcquisitionMode acquisition_mode = AcquisitionMode::XYZRGB;

  enum class PointConversionMode { Custom, OrbbecSDK };
  PointConversionMode conversion_mode = PointConversionMode::Custom;

  enum class ColorResolution { HD_1280x720, FHD_1920x1080 };
  ColorResolution color_resolution = ColorResolution::HD_1280x720;

  int decimation = 1; // @minmax(1, 8)

  TransformConfiguration transform;

  using DeviceType = OrbbecDevice;
  using Tag = rfl::Literal<"orbbec">;
  static constexpr auto PublishPath = "ob";
  static constexpr auto PluginName = "OrbbecDevice";
};

} // namespace pc::devices