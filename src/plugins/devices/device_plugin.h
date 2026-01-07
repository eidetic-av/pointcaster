#pragma once

#include "device_status.h"
#include "device_variants.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <functional>
#include <pointcaster/point_cloud.h>

namespace pc::devices {

class DevicePlugin : public Corrade::PluginManager::AbstractPlugin {
public:
  static Corrade::Containers::StringView pluginInterface() {
    using namespace Corrade::Containers::Literals;
    return "net.pointcaster.DevicePlugin/1.0"_s;
  }

  static Corrade::Containers::Array<Corrade::Containers::String>
  pluginSearchPaths() {
    return {Corrade::InPlaceInit, {"../plugins/devices"}};
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

  // virtual void start() = 0;
  // virtual void stop() = 0;

  // virtual void restart() {
  //   stop();
  //   start();
  // };

  void notify_status_changed(DeviceStatus new_status) {
    if (_status_callback) _status_callback(new_status);
  }

private:
  DeviceConfigurationVariant _config;
  std::function<void(DeviceStatus)> _status_callback;
};

} // namespace pc::devices