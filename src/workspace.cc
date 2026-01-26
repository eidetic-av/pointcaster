#include "workspace.h"
#include "plugins/devices/device_variants.h"
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/Manager.h>
#include <app_settings/app_settings.h>
#include <core/logger/logger.h>
#include <core/uuid/uuid.h>
#include <memory>
#include <metrics/prometheus_server.h>
#include <mutex>
#include <plugins/devices/device_plugin.h>
#include <plugins/plugin_loader.h>
#include <print>
#include <rfl/AddTagsToVariants.hpp>
#include <rfl/json.hpp>
#include <string>
#include <string_view>
#include <thread>
#include <variant>

#ifdef _WIN32
#include <filesystem>
#include <windows.h>
#endif

namespace pc {

using namespace Corrade::PluginManager;
using namespace Corrade::Containers;

bool load_workspace_from_file(WorkspaceConfiguration &config,
                              const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    pc::logger()->error("Could not open '{}'", file_path);
    config = WorkspaceConfiguration{};
    return false;
  }
  const std::string json_string{std::istreambuf_iterator<char>(file),
                                std::istreambuf_iterator<char>()};
  try {
    config = rfl::json::read<WorkspaceConfiguration, rfl::AddTagsToVariants>(
                 json_string)
                 .value();
    pc::logger()->info("Loaded configuration from '{}'", file_path);
    pc::logger()->trace("Loaded Workspace:\n{}",
                      rfl::json::write<rfl::AddTagsToVariants>(config));
  } catch (const std::exception &e) {
    pc::logger()->error("Failed to parse '{}': {}", file_path, e.what());
    config = WorkspaceConfiguration{};
    return false;
  }
  return true;
}

void save_workspace_to_file(const WorkspaceConfiguration &config,
                            const std::string &file_path) {
  try {
    const auto json_string = rfl::json::write<rfl::AddTagsToVariants>(config);
    std::ofstream(file_path) << json_string;
    pc::logger()->info("Saved workspace file to '{}'", file_path);
  } catch (const std::exception &e) {
    pc::logger()->error("Failed to save '{}': {}", file_path, e.what());
  }
}

Workspace::Workspace(const WorkspaceConfiguration &initial) : config(initial) {
  if (initial.id.empty()) {
    config.id = pc::uuid::word();
  } else {
    // if an id was already assigned at workspace initialisation time,
    // we loaded it from disk
    auto_loaded_config = true;
  }

  // start recording metrics
  metrics::PrometheusServer::initialise();

  // find and initialise device plugins
  device_plugin_manager = plugins::load_device_plugins(*this);
  rebuild_devices();

  // TODO
  // ** for debugging
  // print out the current workspace configuration that we have in memory of the
  // workspace simulation
  std::thread([this]() {
    using namespace std::chrono;
    using namespace std::chrono_literals;
    WorkspaceConfiguration thread_config;
    while (true) {
      std::this_thread::sleep_for(5s);
      {
        {
          std::scoped_lock lock(config_access);
          thread_config = config;
        }
        const auto json_string =
            rfl::json::write<rfl::AddTagsToVariants>(thread_config);
        pc::logger()->debug("Workspace configuration: \n{}\n", json_string);
      }
    }
  }).detach();
}

void Workspace::apply_new_config(const WorkspaceConfiguration &new_config) {
  {
    std::scoped_lock lock(config_access);
    config = new_config;
  }
  rebuild_devices();
}

void Workspace::rebuild_devices() {
  devices.clear();
  std::vector<devices::DeviceConfigurationVariant> device_configs;
  {
    std::scoped_lock lock(config_access);
    device_configs = config.devices;
  }
  for (auto &device_config_variant : device_configs) {
    std::visit(
        [&](auto &&device_config) {
          using DeviceConfig = std::decay_t<decltype(device_config)>;
          if (!plugins::is_loaded(*device_plugin_manager,
                                  DeviceConfig::PluginName)) {
            return;
          }
          Pointer<pc::devices::DevicePlugin> device_plugin =
              device_plugin_manager->instantiate(DeviceConfig::PluginName);
          pc::logger()->debug("Updating device plugin config");
          device_plugin->update_config(device_config_variant);
          devices.push_back(std::move(device_plugin));
        },
        device_config_variant);
  }
}

} // namespace pc