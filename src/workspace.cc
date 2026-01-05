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

Workspace::Workspace(WorkspaceConfiguration &config) : config(config) {
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
  revert_config();
}

bool Workspace::loaded_device_plugin(std::string_view plugin_name) const {
  return bool(device_plugin_manager->loadState(plugin_name.data()) &
              LoadState::Loaded);
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
          device_plugin->set_config(device_config_variant);
          devices.push_back(std::move(device_plugin));
        },
        device_config_variant);
  }
}

} // namespace pc