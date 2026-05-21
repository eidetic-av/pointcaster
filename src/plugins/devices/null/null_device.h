#pragma once

#include "../device_plugin.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>

namespace pc::devices {

class NullDevice final : public DevicePlugin {
public:
  explicit NullDevice(Corrade::PluginManager::AbstractManager &manager,
                      Corrade::Containers::StringView plugin)
      : DevicePlugin(manager, plugin) {}

  ~NullDevice() override {}

  NullDevice(const NullDevice &) = delete;
  NullDevice &operator=(const NullDevice &) = delete;
  NullDevice(NullDevice &&) = delete;
  NullDevice &operator=(NullDevice &&) = delete;

  void init(Workspace &workspace) override { _workspace = &workspace; }

  std::vector<DiscoveredDevice> discovered_devices() const override {
    return {};
  };
  void refresh_discovery() override {};
  void add_discovery_change_callback(
      [[maybe_unused]] std::function<void()> cb) override {};
  bool has_discovery_change_callback() const override { return false; };

  bool plugin_null_state() const override { return true; }

  DeviceStatus status() const override { return DeviceStatus::Unloaded; };

  std::shared_ptr<PointCloud> point_cloud() override {
    static auto empty = std::make_shared<PointCloud>(PointCloud{{}, {}, {}});
    return empty;
  };

  void start() override {};
  void stop() override {};
  void restart() override {};
};

} // namespace pc::devices