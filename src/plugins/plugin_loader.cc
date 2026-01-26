#include "plugin_loader.h"
#include "devices/device_plugin.h"

#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/Manager.h>
#include <core/logger/logger.h>
#include <print>
#include <string>
#include <vector>

#ifdef _WIN32
#include <filesystem>
#include <windows.h>
#endif

namespace pc::plugins {

using namespace Corrade::PluginManager;
using namespace Corrade::Containers;

#ifdef _WIN32
namespace {

std::filesystem::path executable_directory_path() {
  wchar_t module_file_path[MAX_PATH];
  const DWORD length = GetModuleFileNameW(nullptr, module_file_path, MAX_PATH);
  if (length == 0 || length == MAX_PATH) {
    throw std::runtime_error("GetModuleFileNameW failed");
  }
  return std::filesystem::path(module_file_path).parent_path();
}

std::filesystem::path default_plugin_root_directory() {
  return executable_directory_path().parent_path() / "plugins";
}

void configure_search_paths(
    const std::filesystem::path &plugin_root_directory) {
  SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                           LOAD_LIBRARY_SEARCH_USER_DIRS);
  AddDllDirectory(plugin_root_directory.wstring().c_str());
  for (const auto &directory_entry :
       std::filesystem::recursive_directory_iterator(plugin_root_directory)) {
    if (!directory_entry.is_directory()) {
      continue;
    }
    const auto directory_path = directory_entry.path();
    const auto wide_path = directory_path.wstring();
    AddDllDirectory(wide_path.c_str());
  }
}

} // namespace
#endif // _WIN32

std::unique_ptr<Manager<devices::DevicePlugin>>
load_device_plugins(pc::Workspace &workspace) {
#ifdef _WIN32
  const auto plugin_root_directory = default_plugin_root_directory();
  configure_search_paths(plugin_root_directory);
#endif

  auto device_plugin_manager =
      std::make_unique<Manager<devices::DevicePlugin>>();

  workspace.loaded_device_plugin_names.clear();

  for (StringView plugin_name : device_plugin_manager->pluginList()) {
    const auto plugin_status = device_plugin_manager->load(plugin_name);
    if (plugin_status & LoadState::Loaded) {
      workspace.loaded_device_plugin_names.push_back(plugin_name);
      pc::logger()->info("Loaded plugin: {}", std::string(plugin_name));

      // create an instance of the plugin that handles device discovery and
      // other static single plugin context things...
      if (!workspace.discovery_plugins.contains(plugin_name)) {
        auto discovery_instance =
            device_plugin_manager->instantiate(plugin_name);
        if (discovery_instance) {
          // start discovery
          discovery_instance->set_is_discovery_instance(true);
          // and pass it over to the workspace that from now on owns the plugin
          // instance
          workspace.discovery_plugins.emplace(std::string(plugin_name),
                                              std::move(discovery_instance));
        }
      }

    }
  }

  return device_plugin_manager;
}

bool is_loaded(Manager<devices::DevicePlugin> &device_plugin_manager,
               std::string_view plugin_name) {
  return bool(device_plugin_manager.loadState(plugin_name.data()) &
              LoadState::Loaded);
}

} // namespace pc::plugins
