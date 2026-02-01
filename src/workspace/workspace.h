#pragma once

#include "metrics/prometheus_server.h"
#include "plugins/devices/device_plugin.h"
#include "plugins/devices/device_variants.h"
#include "workspace_config.h"

#include <Corrade/Containers/Pointer.h>
#include <Corrade/PluginManager/Manager.h>

#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pc {

// Workspace is the application, which can read and write the
// WorkspaceConfiguration.
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

  // updates config and syncs (creates/destroys) device plugin instances
  void apply_new_config(const WorkspaceConfiguration &new_config);

  // sync device plugin instances to match config.devices
  void sync_devices();
};

} // namespace pc
