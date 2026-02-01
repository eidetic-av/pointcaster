#pragma once

#include <plugins/devices/device_variants.h>
#include <session/session_config.h>

namespace pc {

// WorkspaceConfiguration is the config that gets de/serialized and holds all
// simulation state for the running application.
struct WorkspaceConfiguration {
  std::string id;
  std::vector<devices::DeviceConfigurationVariant> devices{};
  std::vector<SessionConfiguration> sessions{};
};

bool load_workspace_from_file(WorkspaceConfiguration &config,
                              const std::string &file_path);

void save_workspace_to_file(const WorkspaceConfiguration &config,
                            const std::string &file_path);

} // namespace pc