#pragma once

#include <functional>
#include <networking/osc/osc_receiver_config.h>
#include <networking/point_streamer_config.h>
#include <optional>
#include <pipeline/concurrent_operator_pipeline_config.h>
#include <plugins/devices/device_group.h>
#include <plugins/devices/device_variants.h>
#include <rfl/DefaultVal.hpp>
#include <session/session_config.h>

namespace pc {

// WorkspaceConfiguration is the config that gets de/serialized and holds all
// simulation state for the running application.
struct WorkspaceConfiguration {
  std::string id;
  rfl::DefaultVal<int> selectedSessionIndex = 0;
  rfl::DefaultVal<int> selectedDeviceIndex = 0;
  std::vector<SessionConfiguration> sessions{};
  std::vector<devices::DeviceConfigurationVariant> devices{};
  std::vector<devices::DeviceGroup> device_groups;
  rfl::DefaultVal<networking::PointStreamerConfiguration> point_streamer;
  rfl::DefaultVal<networking::osc::OscReceiverConfiguration> osc_receiver;
};

bool load_workspace_from_file(WorkspaceConfiguration &config,
                              const std::string &file_path);

void save_workspace_to_file(const WorkspaceConfiguration &config,
                            const std::string &file_path);

std::optional<std::reference_wrapper<SessionConfiguration>>
session_config_from_workspace(const WorkspaceConfiguration &config,
                              std::string_view session_id);

} // namespace pc