#pragma once

#include <config/color_transform_config.h>
#include <config/file_config.h>
#include <config/transform_config.h>
#include <config/sequence_config.h>
#include <plugins/operators/operator_variants.h>
#include <rfl/DefaultVal.hpp>
#include <rfl/Literal.hpp>
#include <string>
#include <vector>

namespace pc::devices {

class PlyDevice;

struct PlyDeviceConfiguration {
  std::string id;     // @hidden
  bool active = true; // @hidden
  bool render = true; // @hidden;

  FileFolderConfiguration file;
  SequenceConfiguration sequence;
  TransformConfiguration transform;
  ColorTransformConfiguration color;

  std::vector<operators::OperatorConfigurationVariant> operators;

  using DeviceType = PlyDevice;
  using Tag = rfl::Literal<"ply">;
  static constexpr auto PluginName = "PlyDevice";
};

} // namespace pc::devices