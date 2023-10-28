#pragma once

#ifndef __CUDACC__
#include "../gui/widgets.h"
#endif

#include "../operators/session_operator_host.h"
#include "../pointer.h"
#include "device_config.h"
#include "driver.h"
#include <Corrade/Containers/Pointer.h>
#include <filesystem>
#include <fstream>
#include <imgui.h>
#include <iterator>
#include <k4abttypes.h>
#include <memory>
#include <mutex>
#include <pointclouds.h>
#include <thread>
#include <variant>
#include <vector>

namespace pc::devices {

enum class DeviceType { UnknownDevice, K4A, K4W2, Rs2 };

class Device {
public:
  static inline std::vector<std::shared_ptr<Device>> attached_devices;
  static inline std::mutex devices_access;

  std::string name = "";
  bool is_sensor = true;
  bool paused = false;

  Device(DeviceConfiguration config);

  virtual std::string id() = 0;

  bool lost_device() { return _driver->lost_device; }

  bool broadcast_enabled() { return _enable_broadcast; }

  auto point_cloud(pc::operators::OperatorList operators = {}) {
    return _driver->point_cloud(_config, operators);
  };

  const DeviceConfiguration config() { return _config; };

  void draw_imgui_controls();

  std::unique_ptr<Driver> _driver;

protected:
  DeviceConfiguration _config;
  bool _enable_broadcast = true;

  // implement this to add device-specific options with imgui
  virtual void draw_device_controls() {}

  const std::string label(std::string label_text, int index = 0) {
    ImGui::Text("%s", label_text.c_str());
    ImGui::SameLine();
    return "##" + name + "_" + _driver->id() + "_" + label_text + "_" +
           std::to_string(index);
  }
};

extern pc::types::PointCloud
synthesized_point_cloud(pc::operators::OperatorList operators = {});

// TODO make all the k4a stuff more generic
using pc::types::Float4;
using K4ASkeleton =
    std::array<std::pair<pc::types::position, Float4>, K4ABT_JOINT_COUNT>;
extern std::vector<K4ASkeleton> scene_skeletons();

extern pc::types::position global_translate;
extern void draw_global_controls();

} // namespace pc::devices
