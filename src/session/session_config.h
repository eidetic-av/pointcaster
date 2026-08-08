#pragma once

#include <camera/camera_config.h>
#include <pipeline/concurrent_operator_pipeline_config.h>
#include <plugins/operators/operator_variants.h>
#include <rfl/DefaultVal.hpp>
#include <string>
#include <vector>

namespace pc {

struct SessionTimelineConfiguration {
  rfl::DefaultVal<bool> looping = true;
  rfl::DefaultVal<int> fps = 30;
  rfl::DefaultVal<int> start_frame = 0;   // @minmax(0, 999999999)
  rfl::DefaultVal<int> current_frame = 0; // @minmax(0, 999999999)
  rfl::DefaultVal<int> end_frame = -1;    // @minmax(-1, 999999999);
  rfl::DefaultVal<int> length = -1;       // @minmax(-1, 999999999);
};

struct SessionConfiguration {
  std::string id;
  rfl::DefaultVal<std::string> label;
  rfl::DefaultVal<CameraConfiguration> camera;
  rfl::DefaultVal<SessionTimelineConfiguration> timeline;
  rfl::DefaultVal<pipeline::ConcurrentOperatorPipelineConfiguration>
      operator_pipeline;
  std::vector<operators::OperatorConfigurationVariant> operators;
};

inline std::string session_address(const SessionConfiguration &config) {
  const auto &label = config.label.value();
  return !label.empty() ? label : config.id;
}

} // namespace pc