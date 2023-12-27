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
    ZoneScopedN("run operators & add result");
    auto device_cloud = device->point_cloud(operators);
    {
      ZoneScopedN("add result");
      //result += std::move(d);
      const auto cloud_size = device_cloud.positions.size();

      result.positions.reserve(result.positions.size() + cloud_size);
      result.positions.insert(
	  result.positions.end(),
	  std::make_move_iterator(device_cloud.positions.begin()),
	  std::make_move_iterator(device_cloud.positions.end()));
      result.colors.reserve(result.colors.size() + cloud_size);

      result.colors.insert(result.colors.end(),
			   std::make_move_iterator(device_cloud.colors.begin()),
			   std::make_move_iterator(device_cloud.colors.end()));
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

  ImGui::SetNextItemOpen(_config.unfolded);

  auto open = _driver->is_open();
  if (!open) ImGui::BeginDisabled();
  
  if (ImGui::CollapsingHeader(name.c_str(), _config.unfolded)) {
    _config.unfolded = true;

    ImGui::TextDisabled("%s", _driver->id().c_str());

    // ImGui::Checkbox("Enable Broadcast", &_enable_broadcast);

    auto running = _driver->is_running();
    if (running) ImGui::BeginDisabled();
    if (ImGui::Button("Start")) {
      // TODO async without being this risky
      std::thread([this] { _driver->start_sensors(); }).detach();
    }
    if (running) ImGui::EndDisabled();
    ImGui::SameLine();
    if (!running) ImGui::BeginDisabled();
    if (ImGui::Button("Stop")) {
      // TODO async without being this risky
      std::thread([this] { _driver->stop_sensors(); }).detach();
    }
    if (!running) ImGui::EndDisabled();

    ImGui::Spacing();

    const bool was_paused = paused;
    ImGui::Checkbox("Pause", &paused);
    if (paused != was_paused) _driver->set_paused(paused);


    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::TreeNode("Alignment")) {
      const bool is_aligning = _driver->is_aligning();
      if (is_aligning)
        ImGui::BeginDisabled();
      if (ImGui::Button(is_aligning ? "Aligning..." : "Start Alignment"))
        _driver->start_alignment();
      if (is_aligning)
        ImGui::EndDisabled();
      bool primary = _driver->primary_aligner;
      if (primary)
        ImGui::BeginDisabled();
      ImGui::Checkbox("Primary", &_driver->primary_aligner);
      if (primary)
        ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled();
      bool aligned = _driver->is_aligned();
      ImGui::Checkbox("Aligned", &aligned);
      ImGui::EndDisabled();
      ImGui::TreePop();
    }

    if (ImGui::TreeNode("Manual Transform")) {
      ImGui::TextDisabled("Flip Input");
      ImGui::Checkbox(label("x", 0).c_str(), &_config.flip_x);
      ImGui::SameLine();
      ImGui::Checkbox(label("y", 0).c_str(), &_config.flip_y);
      ImGui::SameLine();
      ImGui::Checkbox(label("z", 0).c_str(), &_config.flip_z);

      // TODO clean up the following mess into widgets
      // and probably AABBs instead of MinMaxes

      ImGui::TextDisabled("Input Cropping");
      pc::types::MinMax<int> crop_x_in{_config.crop_x.min, _config.crop_x.max};
      pc::types::MinMax<int> crop_y_in{_config.crop_y.min, _config.crop_y.max};
      pc::types::MinMax<int> crop_z_in{_config.crop_z.min, _config.crop_z.max};
      ImGui::SliderInt2(label("x", 1).c_str(), crop_x_in.arr(), -10000, 10000);
      ImGui::SliderInt2(label("y", 1).c_str(), crop_y_in.arr(), -10000, 10000);
      ImGui::SliderInt2(label("z", 1).c_str(), crop_z_in.arr(), -10000, 10000);
      _config.crop_x.min = crop_x_in.min;
      _config.crop_x.max = crop_x_in.max;
      _config.crop_y.min = crop_y_in.min;
      _config.crop_y.max = crop_y_in.max;
      _config.crop_z.min = crop_z_in.min;
      _config.crop_z.max = crop_z_in.max;

      ImGui::TextDisabled("Render Bounds");
      pc::types::MinMax<int> bound_x_out{_config.bound_x.min, _config.bound_x.max};
      pc::types::MinMax<int> bound_y_out{_config.bound_y.min, _config.bound_y.max};
      pc::types::MinMax<int> bound_z_out{_config.bound_z.min, _config.bound_z.max};
      ImGui::SliderInt2(label("x", 2).c_str(), bound_x_out.arr(), -10000, 10000);
      ImGui::SliderInt2(label("y", 2).c_str(), bound_y_out.arr(), -10000, 10000);
      ImGui::SliderInt2(label("z", 2).c_str(), bound_z_out.arr(), -10000, 10000);
      _config.bound_x.min = bound_x_out.min;
      _config.bound_x.max = bound_x_out.max;
      _config.bound_y.min = bound_y_out.min;
      _config.bound_y.max = bound_y_out.max;
      _config.bound_z.min = bound_z_out.min;
      _config.bound_z.max = bound_z_out.max;

      pc::gui::vector_table(id(), "Position", _config.offset, -3000.0f, 3000.0f,
                            0.0f);

      pc::types::Float3 rotation = _config.rotation_deg;
      if (pc::gui::vector_table(id(), "Rotation", rotation, -180.0f, 180.0f,
                                0.0f)) {
        _config.rotation_deg = rotation;
      };

      ImGui::TextDisabled("Scale Output");
      slider(id(), "scale", _config.scale, 0.0f, 4.0f, 1.0f);

      ImGui::TextDisabled("Sample");
      slider(id(), "sample", _config.sample, 1, 50, 1);

      ImGui::TreePop();
    }

    draw_device_controls();
  } else {
    _config.unfolded = false;
  }

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
