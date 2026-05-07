#include "plugin_loader.h"
#include "backend/cpu/cpu_backend.h"
#include "backend/cuda/cuda_backend.h"
#include "devices/device_plugin.h"
#include "devices/null/null_device.h"
#include "operators/operator_plugin.h"

#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/Manager.h>
#include <core/logger/logger.h>
#include <mutex>
#include <print>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <filesystem>
#include <windows.h>
#endif

// import static macros need to be used in global namespace
static void import_static_plugins() {
  static std::once_flag imported_flag;
  std::call_once(imported_flag, [] {
    // these are all the plugins bundled with the pointcaster binary itself
    CORRADE_PLUGIN_IMPORT(CpuBackend)
    CORRADE_PLUGIN_IMPORT(NullDevice)
    CORRADE_PLUGIN_IMPORT(PlyDevice)
  });
}

namespace pc::plugins {

using namespace Corrade::PluginManager;
using namespace Corrade::Containers;

namespace {

#ifdef _WIN32

std::filesystem::path executable_directory_path() {
  wchar_t module_file_path[MAX_PATH];
  const DWORD length = GetModuleFileNameW(nullptr, module_file_path, MAX_PATH);
  if (length == 0 || length == MAX_PATH) {
    throw std::runtime_error("GetModuleFileNameW failed");
  }
  return std::filesystem::path(module_file_path).parent_path();
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

#endif // _WIN32

std::filesystem::path default_plugin_root_directory() {
  return executable_directory_path().parent_path() / "plugins";
}

void configure_plugin_search_path() {
#ifdef _WIN32
  static std::once_flag configured_flag;
  std::call_once(configured_flag, [] {
    const auto plugin_root_directory = default_plugin_root_directory();
    configure_search_paths(plugin_root_directory);
  });
#endif
}

} // namespace

std::unique_ptr<Manager<devices::DevicePlugin>>
load_device_plugins(pc::Workspace &workspace) {

  import_static_plugins();
  configure_plugin_search_path();

  auto device_plugin_manager =
      std::make_unique<Manager<devices::DevicePlugin>>();

  workspace.loaded_device_plugin_names.clear();

  for (StringView plugin_name : device_plugin_manager->pluginList()) {
    const auto plugin_status = device_plugin_manager->load(plugin_name);
    if (plugin_status & LoadState::Loaded) {
      workspace.loaded_device_plugin_names.push_back(plugin_name);

      if (plugin_name == "NullDevice") continue;

      pc::logger()->info("Loaded device plugin '{}'", std::string(plugin_name));

      // create an instance of the plugin that handles device discovery and
      // other static single plugin context things...
      if (!workspace.discovery_plugins.contains(plugin_name)) {
        pc::logger()->trace("Initialising '{}' discovery instance",
                            std::string(plugin_name));
        Corrade::Containers::Pointer<devices::DevicePlugin> discovery_instance;
        try {
          discovery_instance = device_plugin_manager->instantiate(plugin_name);
        } catch (...) {
          pc::logger()->error(
              "{} discovery instance failed to instantiate (unknown exception)",
              std::string(plugin_name));
        }
        if (discovery_instance) {
          // start discovery
          discovery_instance->set_is_discovery_instance(true);
          // and pass it over to the workspace that from now on owns the
          // plugin instance
          workspace.discovery_plugins.emplace(std::string(plugin_name),
                                              std::move(discovery_instance));
          pc::logger()->trace("{} discovery instance added to workspace",
                              std::string(plugin_name));
        } else {
          pc::logger()->error(
              "{} discovery instance failed to instantiate (uncaught error)",
              std::string(plugin_name));
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

std::unique_ptr<Corrade::PluginManager::Manager<backend::BackendPlugin>>
load_backend_plugins(Workspace &workspace) {

  import_static_plugins();
  configure_plugin_search_path();

  auto backend_plugin_manager =
      std::make_unique<Manager<backend::BackendPlugin>>();

  workspace.loaded_backend_plugin_names.clear();

  for (StringView plugin_name : backend_plugin_manager->pluginList()) {
    const auto plugin_status = backend_plugin_manager->load(plugin_name);
    if (plugin_status & LoadState::Loaded) {
      workspace.loaded_backend_plugin_names.push_back(plugin_name);
      pc::logger()->info("Loaded backend plugin '{}'",
                         std::string(plugin_name));
    }
  }

  return backend_plugin_manager;
}

std::unique_ptr<Corrade::PluginManager::Manager<operators::OperatorPlugin>>
load_operator_plugins(Workspace& workspace) {
  import_static_plugins();
  configure_plugin_search_path();

  auto operator_plugin_manager =
      std::make_unique<Manager<operators::OperatorPlugin>>();

  workspace.loaded_operator_plugin_names.clear();

  for (StringView plugin_name : operator_plugin_manager->pluginList()) {
    const auto plugin_status = operator_plugin_manager->load(plugin_name);
    if (plugin_status & LoadState::Loaded) {
      workspace.loaded_operator_plugin_names.push_back(plugin_name);
      pc::logger()->info("Loaded operator plugin '{}'",
                         std::string(plugin_name));
    }
  }

  return operator_plugin_manager;
}

} // namespace pc::plugins
