#pragma once

#include "../point_cloud.h"
#include "../structs.h"
#include "driver.h"
#include <imgui.h>
#include "../gui_helpers.h"
#include <spdlog/spdlog.h>
#include <thread>
#include <fmt/format.h>

namespace bob::sensors {

enum SensorType {
  UnknownDevice,
  K4A,
  Rs2
};

class Device {
public:
  std::string name = "";
  bool flip_x = false;
  bool flip_y = false;
  bool flip_z = false;
  minMax crop_x { -10.f, 10.f };
  minMax crop_y { -10.f, 10.f };
  minMax crop_z { -10.f, 10.f };
  position offset = { 0.f, 0.f, 0.f };
  float scale = 1.f;

  void drawImGuiControls() {

    ImGui::Checkbox("Enable Broadcast", &_enable_broadcast);

    if (ImGui::TreeNode("Transform")) {
        ImGui::TextDisabled("Flip Input");
        ImGui::Checkbox(label("x", 0).c_str(), &flip_x);
        ImGui::SameLine();
        ImGui::Checkbox(label("y", 0).c_str(), &flip_y);
        ImGui::SameLine();
        ImGui::Checkbox(label("z", 0).c_str(), &flip_z);

        ImGui::TextDisabled("Crop Input");
        ImGui::SliderFloat2(label("x", 1).c_str(), crop_x.arr(), -10, 10);
        ImGui::SliderFloat2(label("y", 1).c_str(), crop_y.arr(), -10, 10);
        ImGui::SliderFloat2(label("z", 1).c_str(), crop_z.arr(), -10, 10);

        ImGui::TextDisabled("Offset Output");
        ImGui::SliderFloat(label("x", 2).c_str(), &offset.x, -10, 10);
	gui::enableParameterLearn(&offset.x, gui::ParameterType::Float, -10, 10);
        ImGui::SliderFloat(label("y", 2).c_str(), &offset.y, -10, 10);
	gui::enableParameterLearn(&offset.y, gui::ParameterType::Float, -10, 10);
        ImGui::SliderFloat(label("z", 2).c_str(), &offset.z, -10, 10);
	gui::enableParameterLearn(&offset.z, gui::ParameterType::Float, -10, 10);

        ImGui::TextDisabled("Scale output");
        ImGui::SliderFloat(label("uniform").c_str(), &scale, 0.0f, 3.0f);
	gui::enableParameterLearn(&scale, gui::ParameterType::Float, 0, 3);
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Device options")) {
        drawDeviceSpecificControls();
        ImGui::TreePop();
    }
  };

  bool broadcastEnabled() { return _enable_broadcast; }

  virtual std::string getBroadcastId() = 0;

  bob::PointCloud getPointCloud() {
    DeviceConfiguration config {
      flip_x, flip_y, flip_z,
      crop_x, crop_y, crop_z,
      offset, scale
    };
    return _driver->getPointCloud(config);
  };

protected:
  std::unique_ptr<Driver> _driver;
  bool _enable_broadcast = true;

  // implement this to add device-specific options with imgui
  virtual void drawDeviceSpecificControls() { }

  const std::string label(std::string label_text, int index = 0) {
      ImGui::Text("%s", label_text.c_str());
      ImGui::SameLine();
      return fmt::format("##{}_{}_{}_{}", name, _driver->getId(), label_text, index);
  }
};

} // namespace bob::sensors
