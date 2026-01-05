#include "workspace.h"
#include "plugins/devices/device_plugin.h"
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/Manager.h>
#include <print>
#include <string>
#include <string_view>
#include <variant>

#include <rfl/json.hpp>
#include <string>

namespace pc {

using namespace Corrade::PluginManager;
using namespace Corrade::Containers;

void Workspace::load_config_from_file(WorkspaceConfiguration& config, const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    std::print("Could not open '{}'", file_path);
    config = WorkspaceConfiguration{};
    return;
  }
  const std::string json_string{std::istreambuf_iterator<char>(file),
                                std::istreambuf_iterator<char>()};
  try {
    config = rfl::json::read<WorkspaceConfiguration>(json_string).value();
    std::print("Loaded configuration from '{}'\n", file_path);
  } catch (const std::exception &e) {
    std::print("Failed to parse '{}': {}\n", file_path, e.what());
    config = WorkspaceConfiguration{};
  }
}
Workspace::Workspace(const WorkspaceConfiguration &initial) : config(initial) {
  // find and initialise device plugins
  device_plugin_manager =
      std::make_unique<Manager<pc::devices::DevicePlugin>>();
  std::vector<StringView> loaded_plugin_names{};
  for (auto plugin_name : device_plugin_manager->pluginList()) {
    const auto plugin_status = device_plugin_manager->load(plugin_name);
    if (plugin_status & LoadState::Loaded) {
      loaded_plugin_names.push_back(plugin_name);
      std::println("Loaded plugin: {}", std::string(plugin_name));
    }
  }
  // this is where we could also initailise all plugins that need to run their own thread
  // and then somehow their configuration are synced. we send copies down the pipeline from here though
  revert_config();
}

bool Workspace::loaded_device_plugin(std::string_view plugin_name) const {
  return bool(device_plugin_manager->loadState(plugin_name.data()) &
              LoadState::Loaded);
}

void Workspace::apply_new_config(const WorkspaceConfiguration &new_config) {
  config = new_config;
  revert_config();
}

void Workspace::revert_config() {
  devices.clear();
  for (auto &device_config_variant : config.devices) {
    std::visit(
        [&](auto &&device_config) {
          using DeviceConfig = std::decay_t<decltype(device_config)>;
          if (!loaded_device_plugin(DeviceConfig::PluginName)) {
            return;
          }
          Pointer<pc::devices::DevicePlugin> device_plugin =
              device_plugin_manager->instantiate(DeviceConfig::PluginName);
          device_plugin->update_config(device_config_variant);
          devices.push_back(std::move(device_plugin));
        },
        device_config_variant);
  }
}

} // namespace pc