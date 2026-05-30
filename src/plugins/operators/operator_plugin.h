#pragma once

#include "operator_host.h"
#include "operator_variants.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringStlView.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <Corrade/PluginManager/Manager.h>
#include <Corrade/Tags.h>
#include <camera/camera_frame.h>
#include <cpplocate/cpplocate.h>
#include <filesystem>
#include <logger/logger.h>
#include <pipeline/concurrent_container.h>
#include <plugins/backend/backend_plugin.h>
#include <plugins/backend/backend_types.h>
#include <pointcaster/point_cloud.h>
#include <span>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace pc {
class Workspace;
}

namespace pc::operators {

class OperatorPlugin : public Corrade::PluginManager::AbstractPlugin {
public:
  static Corrade::Containers::StringView pluginInterface() {
    using namespace Corrade::Containers::Literals;
    return "net.pointcaster.OperatorPlugin/1.0"_s;
  }

  static Corrade::Containers::Array<Corrade::Containers::String>
  pluginSearchPaths() {
    std::filesystem::path exe_dir(cpplocate::getModulePath());
    auto plugin_dir = exe_dir.parent_path() / "plugins" / "operators";

    std::vector<Corrade::Containers::String> search_paths;

    for (auto &recursive_file_node :
         std::filesystem::recursive_directory_iterator(plugin_dir)) {
      if (recursive_file_node.path().extension().string() == ".conf") {
        // for each subdirectory that contains a file named .conf,
        // treat it as a plugin directory and add it to our search paths...
        search_paths.emplace_back(
            recursive_file_node.path().parent_path().string());
      }
    }

#ifdef _WIN32
    // convert C:\style\path into posix /style/path used by corrade
    for (auto &path_entry : search_paths) {
      std::string result(path_entry.data());
      result.erase(0, 2);
      std::replace(result.begin(), result.end(), '\\', '/');
      path_entry = result;
    }
#endif

    // return results in a corrade array (required instead of vector)
    static Corrade::Containers::Array<Corrade::Containers::String> results;
    arrayClear(results);
    arrayReserve(results, search_paths.size());
    for (auto &path_string : search_paths) {
      arrayAppend(results, path_string);
    }

    return {Corrade::InPlaceInit, results};
  }

  explicit OperatorPlugin(Corrade::PluginManager::AbstractManager &manager,
                          Corrade::Containers::StringView plugin)
      : Corrade::PluginManager::AbstractPlugin{manager, plugin} {}

  virtual ~OperatorPlugin() = default;

  virtual void init(OperatorHost *host,
                    Corrade::PluginManager::Manager<backend::BackendPlugin>
                        &backend_plugin_manager) {
    _host = host;

    // initialise backends depending on what plugins are available
    using Corrade::PluginManager::LoadState;

    const std::string operator_name{plugin()};

    pc::logger()->trace("Initialising operator plugin: {}", operator_name);

    for (const auto &plugin_name : backend_plugin_manager.pluginList()) {
      const std::string plugin_name_str(plugin_name);
      pc::logger()->debug(plugin_name_str);

      if (backend_plugin_manager.loadState(plugin_name) &
          LoadState::NotLoaded) {
        pc::logger()->debug("{} not loaded!", plugin_name_str);
        continue;
      }
      pc::logger()->debug("{} is loaded", plugin_name_str);
      if (plugin_name == "CpuBackend") {
        _backends[BackendType::CPU] =
            backend_plugin_manager.instantiate(plugin_name);
        _backends[BackendType::CPU]->init();
        pc::logger()->trace("{} created CPU backend", operator_name);
        continue;
      }
      if (plugin_name == "CudaBackend") {
        _backends[BackendType::CUDA] =
            backend_plugin_manager.instantiate(plugin_name);
        _backends[BackendType::CUDA]->init();
        pc::logger()->trace("{} created CUDA backend", operator_name);
      }
    }

    std::visit(
        [this](const auto &c) { set_current_backend(c.backend.value()); },
        config_variant());
  };

  virtual std::shared_ptr<PointCloud> process(const PointCloud &input) = 0;

  void update_config(const OperatorConfigurationVariant &config) {
    _config.store(std::make_shared<const OperatorConfigurationVariant>(config));
  }

  const OperatorConfigurationVariant &config_variant() const {
    return *load_config();
  }

  // if on_config_field_changed needs to be overriden, make sure to call
  // OperatorPlugin::on_config_field_changed as well...
  virtual void on_config_field_changed(std::string_view path = "") {
    if (path == "backend") {
      pc::logger()->error("Backend switch not implemented");
      // TODO
      // std::visit(
      //     [this](const auto &config) {
      //       auto it = _backends.find(config.backend);
      //       if (it != _backends.end()) {
      //         _current_backend = it->second.get();
      //       }
      //     },
      //     _config);
    }
  }

  virtual bool plugin_null_state() const { return false; }

  // if an operator projects any camera / intermediary views into 2d that should
  // be displayed in the UI, declare them as member variables and return them
  // here
  virtual std::vector<camera::CameraFrameRef> camera_frames() { return {}; }

  void set_current_backend(BackendType backend_type) {
    auto it = _backends.find(backend_type);
    if (it != _backends.end()) {
      _current_backend = it->second.get();
    }
  }

protected:
  OperatorHost *_host = nullptr;

  ConcurrentContainer<OperatorConfigurationVariant> _config;

  std::shared_ptr<const OperatorConfigurationVariant> load_config() const {
    return _config.load();
  }

  std::unordered_map<BackendType,
                     Corrade::Containers::Pointer<backend::BackendPlugin>>
      _backends;

  backend::BackendPlugin *_current_backend;
};

} // namespace pc::operators
