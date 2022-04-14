#pragma once

#include "../point_cloud.h"
#include "driver.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <thread>

namespace bob::sensors {

class Device {
public:
  void drawGuiWindow() {
    // spdlog::info("drawGuiWindow");
    ImGui::SetNextWindowPos({50.0f, 50.0f}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.8f);
    ImGui::Begin(_name.c_str(), nullptr);
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.6f);

    ImGui::Checkbox("Enable Broadcast", &_enable_broadcast);

    ImGui::TextDisabled("Crop");
    ImGui::SliderFloat2("x", _driver->crop_x.arr(), -10, 10);
    ImGui::SliderFloat2("y", _driver->crop_y.arr(), -10, 10);
    ImGui::SliderFloat2("z", _driver->crop_z.arr(), -10, 10);

    ImGui::PopItemWidth();
    ImGui::End();
  };

  bool broadcastEnabled() { return _enable_broadcast; }

  virtual bob::PointCloud getPointCloud() = 0;
  virtual std::string getBroadcastId() = 0;

protected:
  std::unique_ptr<Driver> _driver;
  std::string _name = "";
  bool _enable_broadcast = true;
};

} // namespace bob::sensors
