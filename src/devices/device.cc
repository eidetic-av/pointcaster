#include "device.h"
#include "../path.h"
#include "../logger.h"
#include <imgui.h>
#include <k4abttypes.h>
#include <tracy/Tracy.hpp>

#include "k4a/k4a_driver.h"

#ifndef __CUDACC__
#include <zpp_bits.h>
#endif

namespace pc::devices {

using pc::gui::slider;
using pc::operators::OperatorList;

Device::Device(DeviceConfiguration config) : _config(config){};

pc::types::PointCloud
synthesized_point_cloud(OperatorList operators) {
  ZoneScopedN("PointCloud::synthesized_point_cloud");

  if (Device::attached_devices.size() == 0)
    return pc::types::PointCloud{};

  // TODO

  // can run device->point_cloud(operators) in parallel before using them to
  // construct the result

  // For the += operator, return PointCloud& instead of PointCloud to avoid
  // copying the result.

  // it would be good to be able to preallocate memory here?

  pc::types::PointCloud result{};

  std::lock_guard<std::mutex> lock(Device::devices_access);

  for (auto &device : Device::attached_devices) {
    {
      ZoneScopedN("run operators & add result");
      result += device->point_cloud(operators);
    }
  }

  return result;
}

// TODO all of this skeleton stuff needs to be made generic accross multiple
// camera types
std::vector<K4ASkeleton> scene_skeletons() {
  std::vector<K4ASkeleton> result;
  for (auto &device : Device::attached_devices) {
    auto driver = dynamic_cast<K4ADriver *>(device->_driver.get());
    if (!driver->tracking_bodies()) continue;
    for (auto &skeleton : driver->skeletons()) {
      result.push_back(skeleton);
    }
  }
  return result;
}

void Device::draw_imgui_controls() {
  ImGui::PushID(_driver->id().c_str());

  auto open = _driver->is_open();
  if (!open) ImGui::BeginDisabled();

  draw_device_controls();
  if (!open) ImGui::EndDisabled();
  
  ImGui::PopID();
};

pc::types::position global_translate;
void draw_global_controls() {
  ImGui::SetNextWindowPos({150.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::PushID("GlobalTransform");
  ImGui::Begin("Global Transform", nullptr);
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);
  ImGui::PopItemWidth();
  ImGui::End();
  ImGui::PopID();
}

} // namespace pc::devices
