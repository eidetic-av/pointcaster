#pragma once

#include "device_status.h"
#include "device_variants.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <functional>
#include <pointcaster/point_cloud.h>

#ifdef _WIN32
#include <cpplocate/cpplocate.h>
#include <filesystem>
#include <print>
#endif

namespace pc::devices {

class DevicePlugin : public Corrade::PluginManager::AbstractPlugin {
public:
  static Corrade::Containers::StringView pluginInterface() {
    using namespace Corrade::Containers::Literals;
    return "net.pointcaster.DevicePlugin/1.0"_s;
  }

  static Corrade::Containers::Array<Corrade::Containers::String>
  pluginSearchPaths() {
#ifdef _WIN32
    auto exe_dir_canonical = cpplocate::getModulePath();
    std::filesystem::path exe_dir(exe_dir_canonical);
    auto plugin_dir = exe_dir.parent_path() / "plugins";

    // convert C:\style\path into /style/path used by corrade
    const auto to_posix_path = [](std::filesystem::path path) -> std::string {
      auto result = path.string();
      result.erase(0, 2);
      std::replace(result.begin(), result.end(), '\\', '/');
      return result;
    };

    auto devices_plugin_path = to_posix_path(plugin_dir / "devices");
    auto orbbec_plugin_path = to_posix_path(plugin_dir / "devices" / "orbbec");

    return {Corrade::InPlaceInit, { devices_plugin_path, orbbec_plugin_path }};
#else
    return {Corrade::InPlaceInit, {"../plugins/devices"}};
#endif
  }

  explicit DevicePlugin(Corrade::PluginManager::AbstractManager &manager,
                        Corrade::Containers::StringView plugin)
      : Corrade::PluginManager::AbstractPlugin{manager, plugin} {}

  virtual ~DevicePlugin() = default;

  virtual DeviceStatus status() const = 0;

  void set_status_callback(std::function<void(DeviceStatus)> cb) {
    _status_callback = std::move(cb);
  }

  DeviceConfigurationVariant &config() { return _config; }

  void update_config(const DeviceConfigurationVariant &config) {
    _config = config;
  }

  // virtual pc::types::PointCloud point_cloud() const = 0;

  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void restart() = 0;

  void notify_status_changed(DeviceStatus new_status) {
    if (_status_callback) _status_callback(new_status);
  }
  void notify_status_changed() { notify_status_changed(status()); }

private:
  DeviceConfigurationVariant _config;
  std::function<void(DeviceStatus)> _status_callback;
};

} // namespace pc::devices
