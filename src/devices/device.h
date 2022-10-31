#pragma once

#include "../pointer.h"
#include "driver.h"
#include <Corrade/Containers/Pointer.h>
#include <fmt/format.h>
#include <imgui.h>
#include "../gui_helpers.h"
#include <mutex>
#include <pointclouds.h>
#include <thread>
#include <vector>

namespace bob::sensors {

using namespace bob::types;

enum SensorType { UnknownDevice, K4A, K4W2, Rs2 };

class Device {
public:
  std::string name = "";
  bool pause = false;

  bool flip_x = false;
  bool flip_y = false;
  bool flip_z = false;
  minMax<short> crop_x{-10000, 10000};
  minMax<short> crop_y{-10000, 10000};
  minMax<short> crop_z{-10000, 10000};
  short3 offset = {0, 0, 0};
  float3 rotation_deg = {0, 0, 0};
  float scale = 1.f;
  int sample = 1;

  // TODO why is this public
  std::unique_ptr<Driver> _driver;

  virtual std::string getBroadcastId() = 0;

  bool broadcastEnabled() { return _enable_broadcast; }

  bob::types::PointCloud pointCloud() {
    DeviceConfiguration config{flip_x, flip_y, flip_z, crop_x, crop_y,
			       crop_z, offset, rotation_deg, scale, sample};
    return _driver->pointCloud(config);
  };

  uint parameter_index;
  template <typename T>
  void drawSlider(std::string_view label_text, T *value, T min,
		     T max, T default_value = 0) {
    ImGui::PushID(parameter_index++);
    ImGui::Text("%s", label_text.data());
    ImGui::SameLine();
    if constexpr (std::is_integral<T>())
      ImGui::SliderInt(fmt::format("##{}", label_text.data()).c_str(), value, min, max);
    else if constexpr (std::is_floating_point<T>())
      ImGui::SliderFloat(fmt::format("##{}", label_text.data()).c_str(), value, min, max);
    ImGui::SameLine();
    if (ImGui::Button("0")) *value = default_value;
    gui::enableParameterLearn(value, gui::ParameterType::Float, min, max);
    ImGui::PopID();
  }

  void drawImGuiControls() {

    ImGui::PushID(_driver->id().c_str());
    parameter_index = 0;

    ImGui::Checkbox("Enable Broadcast", &_enable_broadcast);

    const bool was_paused = pause;
    ImGui::Checkbox("Pause Sensor", &pause);
    if (pause != was_paused) _driver->setPaused(pause);

    if (ImGui::TreeNode("Alignment")) {
      const bool isAligning = _driver->isAligning();
      if (isAligning) ImGui::BeginDisabled();
      if (ImGui::Button(isAligning ? "Aligning..." : "Start Alignment"))
        _driver->startAlignment();
      if (isAligning) ImGui::EndDisabled();
      bool primary = _driver->primary_aligner;
      if (primary) ImGui::BeginDisabled();
      ImGui::Checkbox("Primary", &_driver->primary_aligner);
      if (primary) ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled();
      bool aligned = _driver->isAligned();
      ImGui::Checkbox("Aligned", &aligned);
      ImGui::EndDisabled();
      ImGui::TreePop();
    }

    if (ImGui::TreeNode("Manual Transform")) {
        ImGui::TextDisabled("Flip Input");
        ImGui::Checkbox(label("x", 0).c_str(), &flip_x);
        ImGui::SameLine();
        ImGui::Checkbox(label("y", 0).c_str(), &flip_y);
        ImGui::SameLine();
        ImGui::Checkbox(label("z", 0).c_str(), &flip_z);

        ImGui::TextDisabled("Crop Input");
	// ImGui can't handle shorts, so we need to use int's then convert
	// back TODO there's probably a better way to do it by defining
	// implicit conversions??
        minMax<int> crop_x_in{crop_x.min, crop_x.max};
        minMax<int> crop_y_in{crop_y.min, crop_y.max};
        minMax<int> crop_z_in{crop_z.min, crop_z.max};
        ImGui::SliderInt2(label("x", 1).c_str(), crop_x_in.arr(), -10000, 10000);
        ImGui::SliderInt2(label("y", 1).c_str(), crop_y_in.arr(), -10000, 10000);
        ImGui::SliderInt2(label("z", 1).c_str(), crop_z_in.arr(), -10000, 10000);
	crop_x.min = crop_x_in.min;
	crop_x.max = crop_x_in.max;
	crop_y.min = crop_y_in.min;
	crop_y.max = crop_y_in.max;
	crop_z.min = crop_z_in.min;
	crop_z.max = crop_z_in.max;

        int3 offset_in{offset.x, offset.y, offset.z};
        ImGui::TextDisabled("Offset Output");
	drawSlider<int>("x", &offset_in.x, -10000, 10000);
	drawSlider<int>("y", &offset_in.y, -10000, 10000);
	drawSlider<int>("z", &offset_in.z, -10000, 10000);
	offset.x = offset_in.x;
	offset.y = offset_in.y;
	offset.z = offset_in.z;

        ImGui::TextDisabled("Rotate Output");
	drawSlider<float>("x", &rotation_deg.x, -180, 180);
	drawSlider<float>("y", &rotation_deg.y, -180, 180);
	drawSlider<float>("z", &rotation_deg.z, -180, 180);

        ImGui::TextDisabled("Scale Output");
	drawSlider<float>("uniform", &scale, 0.0f, 4.0f);

	ImGui::TextDisabled("Sample");
	drawSlider<int>("s", &sample, 1, 10);

        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Device options")) {
        drawDeviceSpecificControls();
        ImGui::TreePop();
    }

    ImGui::PopID();
  };



protected:
  bool _enable_broadcast = true;

  // implement this to add device-specific options with imgui
  virtual void drawDeviceSpecificControls() {}

  const std::string label(std::string label_text, int index = 0) {
    ImGui::Text("%s", label_text.c_str());
    ImGui::SameLine();
    return fmt::format("##{}_{}_{}_{}", name, _driver->id(), label_text, index);
  }
};

// TODO the following free objects and functions should probably be wrapped
// into some class?
extern std::vector<pointer<Device>> attached_devices;
extern std::mutex devices_access;
extern PointCloud synthesizedPointCloud();

} // namespace bob::sensors
