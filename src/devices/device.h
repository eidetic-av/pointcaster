#pragma once

#include "../point_cloud.h"
#include "../structs.h"
#include "driver.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <thread>

namespace bob::sensors {

class Device {
public:
  bool flip_x = false;
  bool flip_y = false;
  bool flip_z = false;
  minMax crop_x { -10.f, 10.f };
  minMax crop_y { -10.f, 10.f };
  minMax crop_z { -10.f, 10.f };
  position offset = { 0.f, 0.f, 0.f };
  float scale = 1.f;

  void drawGuiWindow() {
    // spdlog::info("drawGuiWindow");
    ImGui::SetNextWindowPos({50.0f, 50.0f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.8f);
    ImGui::Begin(_name.c_str(), nullptr);
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.6f);

    ImGui::Checkbox("Enable Broadcast", &_enable_broadcast);

    ImGui::TextDisabled("Flip Input");
    ImGui::Checkbox("x", &flip_x);
    ImGui::SameLine();
    ImGui::Checkbox("y", &flip_y);
    ImGui::SameLine();
    ImGui::Checkbox("z", &flip_z);

    ImGui::TextDisabled("Crop Input");
    ImGui::SliderFloat2("min-max x", crop_x.arr(), -10, 10);
    ImGui::SliderFloat2("min-max y", crop_y.arr(), -10, 10);
    ImGui::SliderFloat2("min-max z", crop_z.arr(), -10, 10);

    ImGui::TextDisabled("Offset Output");
    ImGui::SliderFloat("offset x", &offset.x, -10, 10);
    ImGui::SliderFloat("offset y", &offset.y, -10, 10);
    ImGui::SliderFloat("offset z", &offset.z, -10, 10);

    ImGui::TextDisabled("Scale output");
    ImGui::SliderFloat("overall scale", &scale, 0.0f, 3.0f);

    ImGui::PopItemWidth();
    ImGui::End();
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
  std::string _name = "";
  bool _enable_broadcast = true;
};

} // namespace bob::sensors
