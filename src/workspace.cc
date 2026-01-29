#include "workspace.h"
#include "camera/camera_config.h"
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
#include <rfl/yaml.hpp>
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

// we wrap our workspace configuration structure in a file struct
// so it gets serialized with a top-level "[workspace]" root element
struct WorkspaceFile {
  WorkspaceConfiguration workspace;
};

bool load_workspace_from_file(WorkspaceConfiguration &config,
                              const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    pc::logger()->error("Could not open '{}'", file_path);
    config = WorkspaceConfiguration{};
    return false;
  }
  const std::string yaml_string{std::istreambuf_iterator<char>(file),
                                std::istreambuf_iterator<char>()};
  try {
    config = rfl::yaml::read<WorkspaceFile, rfl::AddTagsToVariants>(yaml_string)
                 .value()
                 .workspace;

    pc::logger()->info("Loaded configuration from '{}'", file_path);
    pc::logger()->trace("Loaded Workspace:\n{}",
                        rfl::yaml::write<rfl::AddTagsToVariants>(
                            WorkspaceFile{.workspace = config}));
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
    const auto yaml_string = rfl::yaml::write<rfl::AddTagsToVariants>(
        WorkspaceFile{.workspace = config});
    std::ofstream(file_path) << yaml_string;
    pc::logger()->info("Saved workspace file to '{}'", file_path);
  } catch (const std::exception &e) {
    pc::logger()->error("Failed to save '{}': {}", file_path, e.what());
  }
}

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
    config.sessions.emplace_back(pc::uuid::word(),
                                 CameraConfiguration{.id = pc::uuid::word()});
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
        const auto yaml_string =
            rfl::yaml::write<rfl::AddTagsToVariants>(thread_config);
        pc::logger()->debug("Workspace configuration: \n{}\n", yaml_string);
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