#pragma once

#include <config/color_transform_config.h>
#include <config/file_config.h>
#include <config/sequence_config.h>
#include <config/transform_config.h>
#include <pipeline/concurrent_operator_pipeline_config.h>
#include <plugins/operators/operator_variants.h>
#include <rfl/DefaultVal.hpp>
#include <rfl/Literal.hpp>
#include <string>
#include <vector>

namespace pc::devices {

class PlyDevice;

struct PlyDeviceConfiguration {
  std::string id; // @hidden

  rfl::DefaultVal<std::string> label;     // @hidden
  rfl::DefaultVal<bool> active = true;    // @hidden
  rfl::DefaultVal<bool> render = true;    // @hidden;
  rfl::DefaultVal<std::string> parent_id; // @hidden
  rfl::DefaultVal<int> order = 0;         // @hidden

  rfl::DefaultVal<FileFolderConfiguration> file;
  rfl::DefaultVal<SequenceConfiguration> sequence;
  rfl::DefaultVal<TransformConfiguration> transform;
  rfl::DefaultVal<ColorTransformConfiguration> color;

  rfl::DefaultVal<operators::ConcurrentOperatorPipelineConfiguration>
      operator_pipeline;                                          // @hidden
  std::vector<operators::OperatorConfigurationVariant> operators; // @hidden

  using DeviceType = PlyDevice;
  using Tag = rfl::Literal<"ply">;
  static constexpr auto PluginName = "PlyDevice";
};

} // namespace pc::devices