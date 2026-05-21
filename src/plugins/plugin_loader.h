#pragma once

#include "operators/operator_plugin.h"
#include <Corrade/PluginManager/Manager.h>
#include <memory>
#include <pointcaster_api.h>
#include <string>
#include <vector>
#include <workspace/workspace.h>

namespace pc::devices {
class DevicePlugin;
}

namespace pc::backend {
class BackendPlugin;
}

namespace pc::plugins {

POINTCASTER_API
    std::unique_ptr<Corrade::PluginManager::Manager<devices::DevicePlugin>>
    load_device_plugins(Workspace &workspace);

POINTCASTER_API
    std::unique_ptr<Corrade::PluginManager::Manager<backend::BackendPlugin>>
    load_backend_plugins(Workspace &workspace);

POINTCASTER_API
    std::unique_ptr<Corrade::PluginManager::Manager<operators::OperatorPlugin>>
    load_operator_plugins(Workspace &workspace);

POINTCASTER_API bool
is_loaded(Corrade::PluginManager::Manager<devices::DevicePlugin>
              &device_plugin_manager,
          std::string_view plugin_name);

POINTCASTER_API bool
is_loaded(Corrade::PluginManager::Manager<backend::BackendPlugin>
              &backend_plugin_manager,
          std::string_view plugin_name);

} // namespace pc::plugins
