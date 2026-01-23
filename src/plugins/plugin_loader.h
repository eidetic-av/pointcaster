#pragma once

#include <Corrade/PluginManager/Manager.h>
#include <memory>
#include <string>
#include <vector>
#include <workspace.h>

namespace pc::devices {
class DevicePlugin;
}

namespace pc::plugins {

std::unique_ptr<Corrade::PluginManager::Manager<devices::DevicePlugin>>
load_device_plugins(Workspace &workspace);

inline std::vector<std::string> loaded_device_plugin_names();

bool is_loaded(Corrade::PluginManager::Manager<devices::DevicePlugin>
                   &device_plugin_manager,
               std::string_view plugin_name);

} // namespace pc::plugins
