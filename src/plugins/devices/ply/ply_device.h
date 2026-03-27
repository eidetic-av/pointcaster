#pragma once

#include "../device_plugin.h"
#include "plugins/devices/device_status.h"
#include "plugins/devices/ply/ply_device_config.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <string_view>

namespace pc::devices {

class PlyDevice : public DevicePlugin {
public:
  explicit PlyDevice(Corrade::PluginManager::AbstractManager &manager,
                     Corrade::Containers::StringView plugin)
      : DevicePlugin(manager, plugin) {}

  ~PlyDevice() override {}

  PlyDevice(const PlyDevice &) = delete;
  PlyDevice &operator=(const PlyDevice &) = delete;
  PlyDevice(PlyDevice &&) = delete;
  PlyDevice &operator=(PlyDevice &&) = delete;

  void init() override;

  DeviceStatus status() const override { return _status; };

  const PointCloud &point_cloud() override { return _current_point_cloud; };

  void start() override {};
  void stop() override {};
  void restart() override {};

  void on_config_field_changed(std::string_view path = "") override;

  bool load_file(std::string_view url);

private:
  std::string _loaded_file_path{};
  DeviceStatus _status = DeviceStatus::Unloaded;

  PointCloud _current_point_cloud{{}, {}};
};

} // namespace pc::devices