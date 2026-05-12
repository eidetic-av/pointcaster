#pragma once

#include "metrics/prometheus_server.h"
#include "plugins/backend/backend_plugin.h"
#include "plugins/devices/device_plugin.h"
#include "plugins/devices/device_variants.h"
#include "plugins/operators/operator_plugin.h"
#include "recorder/session_recorder.h"
#include "workspace_config.h"

#include <Corrade/Containers/Pointer.h>
#include <Corrade/PluginManager/Manager.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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

  std::unique_ptr<Corrade::PluginManager::Manager<backend::BackendPlugin>>
      backend_plugin_manager;
  std::vector<std::string> loaded_backend_plugin_names{};

  std::unique_ptr<Corrade::PluginManager::Manager<devices::DevicePlugin>>
      device_plugin_manager;
  std::vector<std::string> loaded_device_plugin_names{};

  std::vector<Corrade::Containers::Pointer<devices::DevicePlugin>> devices{};
  std::unordered_map<std::string,
                     Corrade::Containers::Pointer<pc::devices::DevicePlugin>>
      discovery_plugins{};

  std::unique_ptr<Corrade::PluginManager::Manager<operators::OperatorPlugin>>
      operator_plugin_manager;
  std::vector<std::string> loaded_operator_plugin_names{};

  std::unique_ptr<metrics::PrometheusServer> prometheus_server;

  std::unique_ptr<recorder::SessionRecorder> session_recorder;

  explicit Workspace(const WorkspaceConfiguration &initial);

  // updates config and syncs (creates/destroys) device plugin instances
  void apply_new_config(const WorkspaceConfiguration &new_config, bool sync_devices = true);

  // sync device plugin instances to match config.devices
  void sync_devices();
};

} // namespace pc
