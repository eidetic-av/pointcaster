#pragma once

#include "plugins/devices/device_plugin.h"
#include "plugins/devices/device_variants.h"

#include <Corrade/Containers/Pointer.h>
#include <Corrade/PluginManager/Manager.h>
#include <memory>
#include <string_view>
#include <vector>

namespace pc {

// WorkspaceConfiguration is the config that gets de/serialized and holds all
// simulation state for the running application.
//
// Workspace is the application, which can read and write the
// WorkspaceConfiguration.

struct WorkspaceConfiguration {
  std::vector<devices::DeviceConfigurationVariant> devices{};
};

void load_workspace_from_file(WorkspaceConfiguration &config,
                              const std::string &file_path);

void save_workspace_to_file(const WorkspaceConfiguration &config,
                            const std::string &file_path);

class Workspace {
public:
  WorkspaceConfiguration config;

  std::unique_ptr<Corrade::PluginManager::Manager<devices::DevicePlugin>>
      device_plugin_manager;
  std::vector<Corrade::Containers::Pointer<devices::DevicePlugin>> devices;

  explicit Workspace(const WorkspaceConfiguration &initial);

  void apply_new_config(const WorkspaceConfiguration &new_config);

  void rebuild_devices();
};

} // namespace pc
