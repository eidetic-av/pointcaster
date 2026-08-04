#include "plugin_loader.h"
#include "backend/cpu/cpu_backend.h"
#include <pointcaster/task_pool.h>
#include "backend/cuda/cuda_backend.h"
#include "devices/device_plugin.h"
#include "devices/null/null_device.h"
#include "operators/operator_plugin.h"

#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/Manager.h>
#include <app_settings/app_settings.h>
#include <core/logger/logger.h>
#include <cpplocate/cpplocate.h>
#include <filesystem>
#include <mutex>
#include <print>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
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
  for (auto directory_entry =
           std::filesystem::recursive_directory_iterator(plugin_root_directory);
       directory_entry != std::filesystem::recursive_directory_iterator();
       ++directory_entry) {
    if (!directory_entry->is_directory()) {
      continue;
    }
    if (directory_entry->path().filename() == "orbbec") {
      // dont recursively add orbbec dependencies
      // (they are handled elsewhere)
      directory_entry.disable_recursion_pending();
      continue;
    }
    const auto directory_path = directory_entry->path();
    const auto wide_path = directory_path.wstring();
    AddDllDirectory(wide_path.c_str());
  }
}

#endif // _WIN32

void configure_plugin_search_path() {
#ifdef _WIN32
  static std::once_flag configured_flag;
  std::call_once(configured_flag, [] {
    // flat windows layout: plugins/ sits beside the executable
    const auto plugin_root_directory = executable_directory_path() / "plugins";
    if (std::filesystem::exists(plugin_root_directory)) {
      configure_search_paths(plugin_root_directory);
    }
  });
#endif
}

// devices plugin root, relative to the executable (differs per platform)
std::filesystem::path device_plugins_root() {
  const std::filesystem::path exe_dir(cpplocate::getModulePath());
#ifdef _WIN32
  // flat windows layout: plugins/ sits beside the executable
  return exe_dir / "plugins" / "devices";
#else
  // linux bin/ layout: plugins/ sits beside the executable's parent dir
  return exe_dir.parent_path() / "plugins" / "devices";
#endif
}

// the orbbec plugin ships one binary per orbbec sdk version (v1 and v2)...
// load the variant matching the application preferences
void load_orbbec_plugin_variant(
    Manager<devices::DevicePlugin> &device_plugin_manager) {
  namespace fs = std::filesystem;

#ifdef _WIN32
  constexpr std::string_view plugin_binary_name = "OrbbecDevice.dll";
#else
  constexpr std::string_view plugin_binary_name = "OrbbecDevice.so";
#endif

  const fs::path orbbec_dir = device_plugins_root() / "orbbec";
  const fs::path sdk_v1_plugin = orbbec_dir / "sdk-v1" / plugin_binary_name;
  const fs::path sdk_v2_plugin = orbbec_dir / "sdk-v2" / plugin_binary_name;

  const bool prefer_v1 = pc::AppSettings::instance()
                             ->value("devices/orbbec/sdkVersion", 2)
                             .toInt() == 1;

  // with only two variants, the other one is always the fallback
  bool using_v1 = prefer_v1;
  if (!fs::exists(using_v1 ? sdk_v1_plugin : sdk_v2_plugin)) {
    using_v1 = !using_v1;
  }
  const fs::path &plugin_path = using_v1 ? sdk_v1_plugin : sdk_v2_plugin;

  if (!fs::exists(plugin_path)) {
    pc::logger()->trace("No orbbec plugin variants found in {}",
                        orbbec_dir.string());
    return;
  }
  if (using_v1 != prefer_v1) {
    pc::logger()->warn("OrbbecDevice sdk v{} variant not found in {}, using "
                       "sdk v{} instead",
                       prefer_v1 ? 1 : 2, orbbec_dir.string(), using_v1 ? 1 : 2);
  }

#ifdef _WIN32
  // make the chosen sdk dlls resolvable
  AddDllDirectory(plugin_path.parent_path().wstring().c_str());
#endif

  // corrade expects utf-8 paths with forward slashes
  const auto load_state =
      device_plugin_manager.load(plugin_path.generic_string());
  if (load_state & LoadState::Loaded) {
    pc::logger()->info("OrbbecDevice plugin is using Orbbec SDK v{}",
                       using_v1 ? 1 : 2);
  } else {
    pc::logger()->error("Failed to load OrbbecDevice plugin from {}",
                        plugin_path.string());
  }
}

} // namespace

std::unique_ptr<Manager<devices::DevicePlugin>>
load_device_plugins(pc::Workspace &workspace) {

  import_static_plugins();
  configure_plugin_search_path();

  auto device_plugin_manager =
      std::make_unique<Manager<devices::DevicePlugin>>();

  // the orbbec plugin is loaded explicitly from the directory
  // matching the preferred orbbec sdk version
  load_orbbec_plugin_variant(*device_plugin_manager);

  workspace.loaded_device_plugin_names.clear();

  for (StringView plugin_name : device_plugin_manager->pluginList()) {
    // skip dependency dlls picked up as plugin candidates without metadata
    if (device_plugin_manager->loadState(plugin_name) &
        LoadState::WrongMetadataFile) {
      continue;
    }
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
    if (backend_plugin_manager->loadState(plugin_name) &
        LoadState::WrongMetadataFile) {
      continue;
    }
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
load_operator_plugins(Workspace &workspace) {
  import_static_plugins();
  configure_plugin_search_path();

  auto operator_plugin_manager =
      std::make_unique<Manager<operators::OperatorPlugin>>();

  workspace.loaded_operator_plugin_names.clear();

  for (StringView plugin_name : operator_plugin_manager->pluginList()) {
    if (operator_plugin_manager->loadState(plugin_name) &
        LoadState::WrongMetadataFile) {
      continue;
    }
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
