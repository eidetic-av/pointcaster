#pragma once

#include <camera/camera_config.h>
#include <pipeline/concurrent_operator_pipeline_config.h>
#include <plugins/operators/operator_variants.h>
#include <rfl/DefaultVal.hpp>
#include <string>
#include <vector>

namespace pc {

struct SessionConfiguration {
  std::string id;
  rfl::DefaultVal<std::string> label;
  rfl::DefaultVal<CameraConfiguration> camera;
  rfl::DefaultVal<pipeline::ConcurrentOperatorPipelineConfiguration>
      operator_pipeline;
  std::vector<operators::OperatorConfigurationVariant> operators;
};

} // namespace pc