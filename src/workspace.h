#pragma once

#include "plugins/devices/device_plugin.h"
#include "plugins/devices/device_variants.h"

#include "metrics/prometheus_server.h"

#include <Corrade/Containers/Pointer.h>
#include <Corrade/PluginManager/Manager.h>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pc {

// WorkspaceConfiguration is the config that gets de/serialized and holds all
// simulation state for the running application.
//
// Workspace is the application, which can read and write the
// WorkspaceConfiguration.

struct WorkspaceConfiguration {
  std::string id;
  std::vector<devices::DeviceConfigurationVariant> devices{};
};

bool load_workspace_from_file(WorkspaceConfiguration &config,
                              const std::string &file_path);

void save_workspace_to_file(const WorkspaceConfiguration &config,
                            const std::string &file_path);

class Workspace {
public:
  WorkspaceConfiguration config;
  std::mutex config_access;

  bool auto_loaded_config = false;

  std::unique_ptr<Corrade::PluginManager::Manager<devices::DevicePlugin>>
      device_plugin_manager;
  std::vector<std::string> loaded_device_plugin_names;
  std::vector<Corrade::Containers::Pointer<devices::DevicePlugin>> devices;
  std::unordered_map<std::string,
                     Corrade::Containers::Pointer<pc::devices::DevicePlugin>>
      discovery_plugins;

  std::unique_ptr<metrics::PrometheusServer> prometheus_server;

  explicit Workspace(const WorkspaceConfiguration &initial);

  void apply_new_config(const WorkspaceConfiguration &new_config);

  void rebuild_devices();
};

} // namespace pc
