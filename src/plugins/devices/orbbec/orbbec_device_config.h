#pragma once

#include <config/network_config.h>
#include <config/transform_config.h>
#include <pointcaster/point_cloud.h>
#include <rfl/Literal.hpp>
#include <string>

namespace pc::devices {

class OrbbecDevice;

struct OrbbecDeviceConfiguration {
  std::string id;     // @hidden
  bool active = true; // @hidden

  std::string ob_uid; // @disabled

  enum class DepthMode { Narrow, Wide };
  DepthMode depth_mode = DepthMode::Narrow;

  enum class AcquisitionMode { XYZRGB, XYZ };
  AcquisitionMode acquisition_mode = AcquisitionMode::XYZRGB;

  enum class PointConversionMode { Custom, OrbbecSDK };
  PointConversionMode conversion_mode = PointConversionMode::Custom;

  enum class ColorResolution {
    HD_1280x720,
    QuadVGA_1280x960,
    FHD_1920x1080,
    QHD_2560x1440,
    UHD_3840x2160
  };
  ColorResolution color_resolution = ColorResolution::HD_1280x720;

  enum class DepthResolution {
    NFOV_320x288,
    WFOV_512x512,
    NFOV_640x576,
    WFOV_1024x1024
  };
  DepthResolution depth_resolution = DepthResolution::NFOV_640x576;

  rfl::Skip<int> fps; // @disabled

  NetworkConfiguration network;

  TransformConfiguration transform;

  using DeviceType = OrbbecDevice;
  using Tag = rfl::Literal<"orbbec">;
  static constexpr auto PublishPath = "ob";
  static constexpr auto PluginName = "OrbbecDevice";
};

namespace orbbec {

using ColorResolution = OrbbecDeviceConfiguration::ColorResolution;
using DepthResolution = OrbbecDeviceConfiguration::DepthResolution;

inline static pc::int2 resolution(ColorResolution resolution_enum) {
  if (resolution_enum == ColorResolution::HD_1280x720) return {1280, 720};
  if (resolution_enum == ColorResolution::FHD_1920x1080) return {1920, 1080};
  return {};
};

inline static pc::int2 resolution(DepthResolution resolution_enum) {
  if (resolution_enum == DepthResolution::NFOV_320x288) return {320, 288};
  if (resolution_enum == DepthResolution::WFOV_512x512) return {512, 512};
  if (resolution_enum == DepthResolution::NFOV_640x576) return {640, 576};
  if (resolution_enum == DepthResolution::WFOV_1024x1024) return {1024, 1024};
  return {};
};

} // namespace orbbec

} // namespace pc::devices