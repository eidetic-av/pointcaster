#pragma once

#include "../gui_helpers.h"
#include "../log.h"
#include "../pointer.h"
#include "../string_utils.h"
#include "driver.h"
#include <Corrade/Containers/Pointer.h>
#include <filesystem>
#include <fstream>
#include <imgui.h>
#include <iterator>
#include <memory>
#include <mutex>
#include <pointclouds.h>
#include <thread>
#include <variant>
#include <vector>

namespace bob::sensors {

using bob::strings::concat;

enum DeviceType { UnknownDevice, K4A, K4W2, Rs2 };

class Device {
public:

  static std::vector<std::shared_ptr<Device>> attached_devices;
  static std::mutex devices_access;
  
  std::string name = "";
  bool is_sensor = true;
  bool paused = false;

  bob::types::DeviceConfiguration config{.flip_x = true,
					 .flip_y = false,
					 .flip_z = true,
					 .crop_x = {-10000, 10000},
					 .crop_y = {-10000, 10000},
					 .crop_z = {-10000, 10000},
					 .offset = {0, -930, 1520},
					 .rotation_deg = {-5, 0, 0},
					 .scale = 1.2f,
					 .sample = 1};

  virtual std::string get_broadcast_id() = 0;

  bool broadcast_enabled() { return _enable_broadcast; }
  auto point_cloud() { return _driver->point_cloud(config); };

  uint _parameter_index;
  template <typename T>
  void draw_slider(std::string_view label_text, T *value, T min, T max,
		  T default_value = 0);

  void draw_imgui_controls();

  void serialize_config() const;
  void deserialize_config(std::vector<uint8_t> data);
  void deserialize_config_from_device_id(const std::string& device_id);
  void deserialize_config_from_this_device();

protected:
  std::unique_ptr<Driver> _driver;
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

extern bob::types::PointCloud synthesized_point_cloud();

extern bob::types::position global_translate;
extern void draw_global_controls();

} // namespace bob::sensors
