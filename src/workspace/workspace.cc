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
  std::vector<devices::DeviceConfigurationVariant> desired_device_configs;
  {
    std::scoped_lock lock(config_access);
    desired_device_configs = config.devices;
  }

  auto device_id_from_variant =
      [](const devices::DeviceConfigurationVariant &v) -> std::string {
    return std::visit([](const auto &cfg) { return cfg.id; }, v);
  };

  auto plugin_name_from_variant =
      [](const devices::DeviceConfigurationVariant &v) -> std::string_view {
    return std::visit(
        [](const auto &cfg) -> std::string_view { return cfg.PluginName; }, v);
  };

  std::unordered_map<std::string, std::size_t> existing_index_by_id;
  existing_index_by_id.reserve(devices.size());

  for (std::size_t i = 0; i < devices.size(); ++i) {
    auto &p = devices[i];
    if (!p) continue;

    const std::string id = device_id_from_variant(p->config());
    if (id.empty()) continue;

    // If duplicates exist, keep the first and let the rest fall through
    // cleanup.
    if (!existing_index_by_id.contains(id)) {
      existing_index_by_id.emplace(id, i);
    }
  }

  std::vector<Pointer<pc::devices::DevicePlugin>> new_devices;
  new_devices.reserve(desired_device_configs.size());

  auto stop_plugin_best_effort = [](pc::devices::DevicePlugin *plugin) {
    if (!plugin) return;
    try {
      plugin->stop();
    } catch (...) {
      // avoid throwing during teardown paths
    }
  };

  for (auto &desired_variant : desired_device_configs) {
    const std::string desired_id = device_id_from_variant(desired_variant);
    if (desired_id.empty()) {
      pc::logger()->warn("Device config missing id; skipping device entry");
      continue;
    }

    const std::string_view desired_plugin_name =
        plugin_name_from_variant(desired_variant);

    if (!plugins::is_loaded(*device_plugin_manager, desired_plugin_name)) {
      pc::logger()->warn(
          "Device plugin '{}' not loaded; skipping device id='{}'",
          std::string(desired_plugin_name), desired_id);
      continue;
    }

    Pointer<pc::devices::DevicePlugin> keep_or_new;

    auto it = existing_index_by_id.find(desired_id);
    if (it != existing_index_by_id.end()) {
      // Reuse existing plugin instance for this id if variant type matches.
      const std::size_t existing_index = it->second;
      existing_index_by_id.erase(it);

      if (existing_index < devices.size() && devices[existing_index]) {
        auto &existing_ptr = devices[existing_index];
        auto *existing_plugin = existing_ptr.get();

        const std::string_view existing_plugin_name =
            plugin_name_from_variant(existing_plugin->config());

        const bool plugin_type_matches =
            (existing_plugin_name == desired_plugin_name);

        if (plugin_type_matches) {
          // Update config in-place (does not restart pipelines by itself).
          existing_plugin->update_config(desired_variant);
          keep_or_new = std::move(existing_ptr);
        } else {
          // Same id but different variant type: replace plugin instance.
          pc::logger()->warn("Device id='{}' changed plugin type '{}' -> '{}'; "
                             "replacing plugin",
                             desired_id, std::string(existing_plugin_name),
                             std::string(desired_plugin_name));

          stop_plugin_best_effort(existing_plugin);
          existing_ptr = nullptr;

          keep_or_new =
              device_plugin_manager->instantiate(desired_plugin_name.data());
          if (keep_or_new) {
            keep_or_new->set_is_discovery_instance(false);
            keep_or_new->update_config(desired_variant);
          }
        }
      }
    } else {
      // New id: instantiate fresh.
      keep_or_new =
          device_plugin_manager->instantiate(desired_plugin_name.data());
      if (keep_or_new) {
        keep_or_new->set_is_discovery_instance(false);
        keep_or_new->update_config(desired_variant);
      }
    }

    if (keep_or_new) {
      new_devices.push_back(std::move(keep_or_new));
    }
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

  // Replace with the new ordered list.
  devices = std::move(new_devices);
}

} // namespace pc
