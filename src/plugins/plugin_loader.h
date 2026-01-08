#pragma once

#include <Corrade/PluginManager/Manager.h>
#include <memory>

namespace pc::devices {
class DevicePlugin;
}

namespace pc::plugins {

std::unique_ptr<Corrade::PluginManager::Manager<devices::DevicePlugin>>
load_device_plugins();

bool is_loaded(Corrade::PluginManager::Manager<devices::DevicePlugin>
                   &device_plugin_manager,
               std::string_view plugin_name);

} // namespace pc::plugins
