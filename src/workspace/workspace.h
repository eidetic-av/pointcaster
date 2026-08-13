#pragma once

#include "config/config_registry.h"
#include "session/session.h"
#include "workspace_config.h"

#include <Corrade/Containers/Pointer.h>
#include <Corrade/PluginManager/Manager.h>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <plugins/backend/backend_plugin.h>
#include <plugins/devices/device_plugin.h>
#include <plugins/devices/device_variants.h>
#include <plugins/operators/operator_plugin.h>
#include <stop_token>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pc {

namespace recorder {
class SessionRecorder;
}

namespace networking {
class PointStreamer;
}

namespace publishers {
class WorkspacePublisher;
class MqttClient;
class OscSender;
} // namespace publishers

namespace receivers {
class OscReceiver;
}

// Workspace is the application, which can read and write the
// WorkspaceConfiguration.
class Workspace {
public:
  WorkspaceConfiguration config;
  std::mutex config_access;

  pc::ConfigRegistry config_registry;

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

  std::unique_ptr<recorder::SessionRecorder> session_recorder;
  std::unique_ptr<networking::PointStreamer> point_streamer;

  std::unique_ptr<publishers::WorkspacePublisher> workspace_publisher;
  std::unique_ptr<publishers::MqttClient> mqtt_client;
  std::unique_ptr<publishers::OscSender> osc_sender;

  std::unique_ptr<receivers::OscReceiver> osc_receiver;

  std::unordered_map<std::string, std::unique_ptr<Session>> sessions;

  explicit Workspace(const WorkspaceConfiguration &initial);
  // defined in workspace.cc, where the forward-declared members are complete
  ~Workspace();

  // updates config and syncs (creates/destroys) device plugin instances
  void apply_new_config(const WorkspaceConfiguration &new_config,
                        bool sync_devices = true);

  // sync device plugin instances to match config.devices
  void sync_devices();

  void sync_sessions();

  void rebuild_config_registry();

private:
  void metrics_thread_work(std::stop_token stop_token);
  std::jthread _metrics_thread;
};

} // namespace pc
