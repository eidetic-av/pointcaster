#pragma once

#include "../device_plugin.h"

#include "plugins/devices/device_status.h"
#include "plugins/devices/ply/ply_device_config.h"
#include "ply_device_config.h"
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <plugins/backend/backend_plugin.h>
#include <readerwriterqueue/readerwritercircularbuffer.h>
#include <string_view>
#include <vector>

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

  void init(Workspace &workspace) override;

  DeviceStatus status() const override { return _status; };

  std::shared_ptr<PointCloud> point_cloud() override;

  void start() override {};
  void stop() override {};
  void restart() override {};

  void on_config_field_changed(std::string_view path = "") override;

  bool load_file(std::string_view url);

private:
  std::string _loaded_file_path{};
  DeviceStatus _status = DeviceStatus::Unloaded;

  std::shared_ptr<PointCloud> _input_cloud;

  Corrade::Containers::Pointer<backend::BackendPlugin> _cpu_backend;
  Corrade::Containers::Pointer<backend::BackendPlugin> _cuda_backend;

  std::shared_ptr<PointCloud> _current_point_cloud =
      std::make_shared<PointCloud>(PointCloud{{}, {}});

  void apply_transform();
};

} // namespace pc::devices