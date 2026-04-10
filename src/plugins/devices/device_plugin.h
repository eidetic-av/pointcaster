#pragma once

#include "device_status.h"
#include "device_variants.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <Corrade/Tags.h>
#include <cpplocate/cpplocate.h>
#include <filesystem>
#include <functional>
#include <logger/logger.h>
#include <pointcaster/point_cloud.h>
#include <string>

namespace pc {
class Workspace;
}

namespace pc::devices {

struct DiscoveredDevice {
  std::string label;
  std::string ip;
  std::string id;
};

class DevicePlugin : public Corrade::PluginManager::AbstractPlugin {
public:
  static Corrade::Containers::StringView pluginInterface() {
    using namespace Corrade::Containers::Literals;
    return "net.pointcaster.DevicePlugin/1.0"_s;
  }

  static Corrade::Containers::Array<Corrade::Containers::String>
  pluginSearchPaths() {
    std::filesystem::path exe_dir(cpplocate::getModulePath());
    auto plugin_dir = exe_dir.parent_path() / "plugins" / "devices";

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

  explicit DevicePlugin(Corrade::PluginManager::AbstractManager &manager,
                        Corrade::Containers::StringView plugin)
      : Corrade::PluginManager::AbstractPlugin{manager, plugin} {
    static bool loaded_discovery_instance = false;
    if (!loaded_discovery_instance) {
      _is_discovery_instance = true;
      loaded_discovery_instance = true;
    }
  }

  virtual ~DevicePlugin() = default;

  virtual void init(Workspace &workspace) = 0;

  void set_is_discovery_instance(bool v) { _is_discovery_instance = v; }
  bool is_discovery_instance() { return _is_discovery_instance; };

  virtual std::vector<DiscoveredDevice> discovered_devices() const {
    return {};
  };
  virtual void refresh_discovery() {};

  virtual void add_discovery_change_callback(std::function<void()>){};
  virtual bool has_discovery_change_callback() const { return false; };

  virtual DeviceStatus status() const = 0;

  void set_status_callback(std::function<void(DeviceStatus)> cb) {
    _status_callback = cb;
  }

  void set_point_cloud_updated_callback(std::function<void()> cb) {
    _point_cloud_updated_callback = cb;
  }

  DeviceConfigurationVariant &config() { return _config; }

  void update_config(const DeviceConfigurationVariant &config) {
    _config = config;
  }

  virtual void on_config_field_changed(std::string_view path = "") {}

  virtual bool plugin_null_state() const { return false; }

  virtual const PointCloud &point_cloud() = 0;

  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void restart() = 0;

  void notify_status_changed(DeviceStatus new_status) {
    if (_status_callback) _status_callback(new_status);
  }
  void notify_status_changed() { notify_status_changed(status()); }

  void notify_point_cloud_updated() {
    if (_point_cloud_updated_callback) _point_cloud_updated_callback();
  }

protected:
  Workspace *_workspace;
  DeviceConfigurationVariant _config;
  std::function<void(DeviceStatus)> _status_callback;
  std::function<void()> _point_cloud_updated_callback;
  bool _is_discovery_instance = false;
};

} // namespace pc::devices
