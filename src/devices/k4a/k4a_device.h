#pragma once

#include "../device.h"
#include "k4a_config.gen.h"
#include "k4a_driver.h"
#include <array>
#include <atomic>
#include <k4abt.hpp>
#include <utility>
#include <vector>

namespace pc::devices {

class K4ADevice : public DeviceBase<AzureKinectConfiguration> {
public:
  inline static std::vector<std::reference_wrapper<K4ADevice>> attached_devices;
  inline static std::mutex devices_access;

  explicit K4ADevice(AzureKinectConfiguration &config);
  ~K4ADevice();

  K4ADevice(const K4ADevice &) = delete;
  K4ADevice &operator=(const K4ADevice &) = delete;
  K4ADevice(K4ADevice &&) = delete;
  K4ADevice &operator=(K4ADevice &&) = delete;

  std::unique_ptr<Driver> _driver;
  std::string name = "";
  bool is_sensor = true;
  bool paused = false;
  bool lost_device() const { return _driver->lost_device; }
  bool broadcast_enabled() const { return _enable_broadcast; }

  DeviceStatus status() const override;
  pc::types::PointCloud point_cloud() override {
    return _driver->point_cloud(this->config());
  };

  void draw_imgui_controls();
  AzureKinectConfiguration _config;
  bool _enable_broadcast = true;

  const std::string label(std::string label_text, int index = 0) {
    ImGui::Text("%s", label_text.c_str());
    ImGui::SameLine();
    return "##" + name + "_" + _driver->id() + "_" + label_text + "_" +
           std::to_string(index);
  }

  void draw_device_controls();
  void update_device_control(int *target, int value,
                             std::function<void(int)> set_func);

  void reattach(int index);

  static inline std::atomic<std::size_t> count;
  static std::string get_serial_number(const std::size_t device_index);

  static inline std::size_t active_driver_count() {
    return K4ADriver::active_count;
  };
};

} // namespace pc::devices
