#pragma once
#include "plugins/devices/device_plugin.h"
#include "plugins/devices/device_variants.h"
#include <Corrade/Containers/Pointer.h>
#include <memory>
#include <vector>

namespace pc {

// WorkspaceConfiguration is the config that gets de/serialized and holds all
// simulation state for the running application.

// Workspace is the application, which can read and write the
// WorkspaceConfiguration

struct WorkspaceConfiguration {
  std::vector<devices::DeviceConfigurationVariant> devices;
};

void load_workspace(WorkspaceConfiguration &config, std::string_view path);

class Workspace {
public:

  std::unique_ptr<Corrade::PluginManager::Manager<devices::DevicePlugin>>
      device_plugin_manager;
  std::vector<Corrade::Containers::Pointer<devices::DevicePlugin>> devices;

  Workspace(WorkspaceConfiguration &config);

  bool loaded_device_plugin(std::string_view plugin_name) const;

  void revert_config();

private:
  WorkspaceConfiguration &_config;
};

} // namespace pc