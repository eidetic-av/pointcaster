#pragma once

#include "device_variants.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <pointcaster/point_cloud.h>

namespace pc::devices {

enum class DeviceStatus { Loaded, Active, Inactive, Missing };

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

  DeviceConfigurationVariant &config() { return _config; }

  void update_config(const DeviceConfigurationVariant &config) {
    _config = config;
  }

  // virtual DeviceStatus status() const = 0;

  // virtual pc::types::PointCloud point_cloud() const = 0;

  // virtual void start() = 0;
  // virtual void stop() = 0;

  // virtual void restart() {
  //   stop();
  //   start();
  // };

  private:
    DeviceConfigurationVariant _config;
};

} // namespace pc::devices