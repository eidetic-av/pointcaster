#include "plugin_loader.h"
#include "devices/device_plugin.h"

#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/Manager.h>
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
  return executable_directory_path() / "../plugins";
}

void configure_search_paths(
    const std::filesystem::path &plugin_root_directory) {
  SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                           LOAD_LIBRARY_SEARCH_USER_DIRS);

  const auto device_plugin_directory = plugin_root_directory / "devices";

  if (!std::filesystem::exists(device_plugin_directory)) {
    return;
  }

  for (const auto &directory_entry :
       std::filesystem::directory_iterator(device_plugin_directory)) {
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

std::unique_ptr<Manager<devices::DevicePlugin>> load_device_plugins() {
#ifdef _WIN32
  const auto plugin_root_directory = default_plugin_root_directory();
  configure_search_paths(plugin_root_directory);
#endif

  auto device_plugin_manager =
      std::make_unique<Manager<devices::DevicePlugin>>();

  std::vector<StringView> loaded_plugin_names;

  for (StringView plugin_name : device_plugin_manager->pluginList()) {
    const auto plugin_status = device_plugin_manager->load(plugin_name);
    if (plugin_status & LoadState::Loaded) {
      loaded_plugin_names.push_back(plugin_name);
      std::println("Loaded plugin: {}", std::string(plugin_name));
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
