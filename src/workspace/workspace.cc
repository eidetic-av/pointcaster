#include "workspace.h"

#include "camera/camera_config.h"
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

  // start recording metrics
  metrics::PrometheusServer::initialise();

  // find and initialise device plugins
  device_plugin_manager = plugins::load_device_plugins(*this);

  // instantiate device plugins for the initial config
  sync_devices();
}

void Workspace::apply_new_config(const WorkspaceConfiguration &new_config) {
  {
    std::scoped_lock lock(config_access);
    config = new_config;
  }
  sync_devices();
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

    const auto [id, _] = device_info_from_variant(p->config());
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
    bool new_device_instance = false;

    auto it = existing_index_by_id.find(device_id);
    if (it != existing_index_by_id.end()) {
      // Reuse existing plugin instance for this id if variant type matches.
      const std::size_t existing_index = it->second;
      existing_index_by_id.erase(it);

      if (existing_index < devices.size() && devices[existing_index]) {
        auto &existing_ptr = devices[existing_index];
        auto *existing_plugin = existing_ptr.get();

        const auto [_, existing_plugin_name] =
            device_info_from_variant(existing_plugin->config());

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

          if (plugin_loaded) {
            device_plugin =
                device_plugin_manager->instantiate(device_plugin_name);
            new_device_instance = true;
          } else {
            device_plugin = device_plugin_manager->instantiate("NullDevice");
          }
          device_plugin->set_is_discovery_instance(false);
          device_plugin->update_config(device_variant);
        }
      }
    } else {
      if (plugin_loaded) {
        pc::logger()->trace("Instantiating a new device plugin instance");
        device_plugin = device_plugin_manager->instantiate(device_plugin_name);
        new_device_instance = true;
      }
      if (device_plugin) {
        std::visit(
            [](auto &&config) {
              pc::logger()->debug("Applying device configuration: {}",
                                  config.ip);
            },
            device_variant);
        device_plugin->set_is_discovery_instance(false);
        device_plugin->update_config(device_variant);
      }
    }
    if (new_device_instance) device_plugin->init();
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
