#include "workspace.h"

#include "camera/camera_config.h"
#include "config/config_registry.h"
#include "networking/osc/osc_receiver.h"
#include "networking/point_streamer.h"
#include "session/session.h"
#include "session/session_config.h"
#include "workspace_config.h"

#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/Manager.h>

#include <app_settings/app_settings.h>
#include <core/logger/logger.h>
#include <core/uuid/uuid.h>
#include <metrics/prometheus_server.h>
#include <plugins/devices/device_plugin.h>
#include <plugins/devices/device_variants.h>
#include <plugins/plugin_loader.h>

#include <memory>
#include <mutex>
#include <ranges>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <filesystem>
#include <windows.h>
#endif

namespace pc {

using namespace Corrade::PluginManager;
using namespace Corrade::Containers;

Workspace::Workspace(const WorkspaceConfiguration &initial) : config(initial) {
  if (config.id.empty()) {
    config.id = pc::uuid::word();
  } else {
    // if an id was already assigned at workspace initialisation time,
    // we loaded it from disk
    auto_loaded_config = true;
  }

  if (config.sessions.empty()) {
    // every workspace needs a session with a camera
    config.sessions.emplace_back(pc::uuid::word(), "session_1",
                                 CameraConfiguration{.id = pc::uuid::word()});
  }

  // find and initialise plugins
  backend_plugin_manager = plugins::load_backend_plugins(*this);
  device_plugin_manager = plugins::load_device_plugins(*this);
  operator_plugin_manager = plugins::load_operator_plugins(*this);

  // instantiate device plugins for the initial config
  sync_devices();

  // instantiate session operator pipelines
  sync_sessions();

  rebuild_config_registry();

  // initialise workspace instances
  osc_receiver = std::make_unique<networking::osc::OscReceiver>(*this);
  session_recorder = std::make_unique<recorder::SessionRecorder>(*this);
  point_streamer = std::make_unique<networking::PointStreamer>(*this);

  // TODO maybe the metrics server shouldn't be a singleton and should
  // follow the same pattern as session_recorder & point_streamer belonging
  // to this workspace class and the injected workspace is what grabs it
  // wherever it's needed... at least just so it can be torn down and restarted
  // at will
  metrics::PrometheusServer::initialise();
}

void Workspace::apply_new_config(const WorkspaceConfiguration &new_config,
                                 bool should_sync_devices) {
  {
    std::scoped_lock lock(config_access);
    config = new_config;
  }
  if (should_sync_devices) sync_devices();
  sync_sessions();
  rebuild_config_registry();
  if (osc_receiver) osc_receiver->reconfigure();
}

// TODO not sure about this...
// could probs be comp time registry? idk at least
// it probs doesn't need to be rebuilt often
void Workspace::rebuild_config_registry() {
  config_registry.clear();
  for (auto &session_config : config.sessions) {
    pc::register_config(config_registry, "session/" + session_config.id,
                        session_config);
  }
  for (auto &device_plugin : devices) {
    if (!device_plugin) continue;
    std::visit(
        [this](auto &device_config) {
          pc::register_config(config_registry, "device/" + device_config.id,
                              device_config);
        },
        device_plugin->config_variant());
  }
  pc::register_config(config_registry, "streaming",
                      config.point_streamer.value());
  pc::register_config(config_registry, "osc", config.osc_receiver.value());
}

void Workspace::sync_sessions() {
  pc::logger()->trace("Syncing sessions");

  std::vector<SessionConfiguration> session_configs;
  {
    std::scoped_lock lock(config_access);
    session_configs = config.sessions;
  }

  // create any new sessions, and push updated config into existing ones so
  // their operator set / pipeline reflects the latest configuration
  for (auto &session_config : session_configs) {
    auto it = sessions.find(session_config.id);
    if (it == sessions.end()) {
      sessions.emplace(session_config.id,
                       std::make_unique<Session>(*this, session_config));
    } else if (it->second) {
      it->second->update_config(session_config);
    }
  }

  // erase any removed sessions
  std::vector<std::string> sessions_to_erase;
  for (auto &[session_id, session] : sessions) {
    auto it = std::ranges::find(session_configs, session_id,
                                &SessionConfiguration::id);
    if (!session || it == session_configs.end())
      sessions_to_erase.emplace_back(session_id);
  }
  for (auto &session_id : sessions_to_erase) {
    sessions.erase(session_id);
  }
}

void Workspace::sync_devices() {
  pc::logger()->trace("Syncing devices");
  std::vector<devices::DeviceConfigurationVariant> device_configs;
  {
    std::scoped_lock lock(config_access);
    device_configs = config.devices;
  }

  // check for duplicates
  std::unordered_map<std::string, std::size_t> existing_index_by_id;
  existing_index_by_id.reserve(devices.size());

  for (std::size_t i = 0; i < devices.size(); ++i) {
    auto &p = devices[i];
    if (!p) continue;

    const auto [id, _] = device_info_from_variant(p->config_variant());
    if (id.empty()) continue;

    // If duplicates exist, keep the first
    if (!existing_index_by_id.contains(id)) {
      existing_index_by_id.emplace(id, i);
    }
  }

  // the result collection of device plugins
  std::vector<Pointer<pc::devices::DevicePlugin>> new_devices;
  new_devices.reserve(device_configs.size());

  auto stop_plugin_best_effort = [](pc::devices::DevicePlugin *plugin) {
    if (!plugin) return;
    try {
      plugin->stop();
    } catch (...) {
      // avoid throwing during teardown paths
    }
  };

  for (auto &device_variant : device_configs) {

    auto [device_id, device_plugin_name] =
        device_info_from_variant(device_variant);

    if (device_id.empty()) {
      pc::logger()->warn("Device config missing id; skipping device entry");
      continue;
    }

    bool plugin_loaded =
        plugins::is_loaded(*device_plugin_manager, device_plugin_name);

    if (!plugin_loaded) {
      pc::logger()->error(
          "Device plugin '{}' not loaded when syncing device '{}'",
          device_plugin_name, device_id);
      // treat the device plugin as NullDevice
      device_plugin_name = "NullDevice";
      plugin_loaded = true;
    }

    Pointer<pc::devices::DevicePlugin> device_plugin = nullptr;

    auto it = existing_index_by_id.find(device_id);
    if (it != existing_index_by_id.end()) {
      // Reuse existing plugin instance for this id if variant type matches.
      const std::size_t existing_index = it->second;
      existing_index_by_id.erase(it);

      if (existing_index < devices.size() && devices[existing_index]) {
        auto &existing_ptr = devices[existing_index];
        auto *existing_plugin = existing_ptr.get();

        const auto [_, existing_plugin_name] =
            device_info_from_variant(existing_plugin->config_variant());

        if (existing_plugin_name == device_plugin_name) {
          // Update config in-place (does not restart pipelines by itself).
          existing_plugin->update_config(device_variant);
          device_plugin = std::move(existing_ptr);
        } else {
          // Same id but different variant type (device type changed), so
          // replace plugin instance
          pc::logger()->info("Device id='{}' changed plugin type '{}' -> '{}'",
                             device_id, existing_plugin_name,
                             device_plugin_name);

          stop_plugin_best_effort(existing_plugin);
          existing_ptr = nullptr;
          bool new_device_instance = false;

          if (plugin_loaded) {
            device_plugin =
                device_plugin_manager->instantiate(device_plugin_name);
            new_device_instance = true;
          } else {
            device_plugin = device_plugin_manager->instantiate("NullDevice");
            new_device_instance = true;
          }
          device_plugin->set_is_discovery_instance(false);
          if (new_device_instance) device_plugin->init(*this);
          device_plugin->update_config(device_variant);
        }
      }
    } else {
      if (plugin_loaded) {
        pc::logger()->trace("Instantiating a new device plugin instance");
        device_plugin = device_plugin_manager->instantiate(device_plugin_name);
      }
      if (device_plugin) {
        std::visit(
            [](auto &&config) {
              pc::logger()->trace("Applying device configuration: {}",
                                  config.id);
            },
            device_variant);
        device_plugin->set_is_discovery_instance(false);
        device_plugin->init(*this);
        device_plugin->update_config(device_variant);
      }
    }
    new_devices.push_back(std::move(device_plugin));
  }

  // Remaining entries in existing_index_by_id are deletions.
  // We own them in `devices` (until we overwrite below), so stop best-effort
  // now.
  for (const auto &[id, idx] : existing_index_by_id) {
    if (idx >= devices.size()) continue;
    if (!devices[idx]) continue;
    pc::logger()->trace("Removing device plugin id='{}'", id);
    stop_plugin_best_effort(devices[idx].get());
    devices[idx] = nullptr;
  }

  devices = std::move(new_devices);
}

} // namespace pc