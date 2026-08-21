#pragma once

#include <camera/camera_config.h>
#include <pipeline/concurrent_operator_pipeline_config.h>
#include <plugins/operators/operator_variants.h>
#include <rfl/DefaultVal.hpp>
#include <set>
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
  std::string id;                                         // @hidden
  rfl::DefaultVal<std::string> label;                     // @hidden
  rfl::DefaultVal<CameraConfiguration> camera;            // @folded
  rfl::DefaultVal<SessionTimelineConfiguration> timeline; // @hidden
  rfl::DefaultVal<operators::ConcurrentOperatorPipelineConfiguration>
      operator_pipeline;                                          // @folded
  std::vector<operators::OperatorConfigurationVariant> operators; // @hidden
  rfl::DefaultVal<std::set<std::string>> disabled_devices;        // @hidden
};

inline std::string session_address(const SessionConfiguration &config) {
  const auto &label = config.label.value();
  return !label.empty() ? label : config.id;
}

} // namespace pc
