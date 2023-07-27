#include "device.h"
#include "../path.h"
#include <spdlog/spdlog.h>
#include <imgui.h>

#ifndef __CUDACC__
#include <zpp_bits.h>
#endif

namespace pc::sensors {

std::mutex Device::devices_access;
std::vector<std::shared_ptr<Device>> Device::attached_devices;

pc::types::PointCloud synthesized_point_cloud() {
  auto result = pc::types::PointCloud{};
  if (Device::attached_devices.size() == 0)
    return result;
  // std::lock_guard<std::mutex> lock(devices_access);
  for (auto &device : Device::attached_devices)
    result += device->point_cloud();
  return result;
}

void Device::serialize_config() const {
  // TODO error handling
  spdlog::info("Serializing device configuration for '{}'", this->name);
  auto id = this->_driver->id();
  std::vector<uint8_t> data;
  auto out = zpp::bits::out(data);
  auto success = out(config);
  auto data_dir = path::get_or_create_data_directory();
  auto file_path = data_dir / (id + ".pcc");
  spdlog::info("Saving to {}", file_path.string());
  path::save_file(file_path, data);
};

void Device::deserialize_config(std::vector<uint8_t> buffer) {
  auto in = zpp::bits::in(buffer);
  auto success = in(this->config);
}

void Device::deserialize_config_from_device_id(const std::string &device_id) {
  auto data_dir = path::data_directory();
  auto file_path = data_dir / (device_id + ".pcc");
  spdlog::info("Loading from {}", file_path.string());

  if (!std::filesystem::exists(file_path)) {
    spdlog::warn("Config doesn't exist");
    return;
  }

  deserialize_config(path::load_file(file_path));
  spdlog::info("Loaded");
}

void Device::deserialize_config_from_this_device() {
  deserialize_config_from_device_id(_driver->id());
}

template <typename T>
void Device::draw_slider(std::string_view label_text, T *value, T min, T max,
                         T default_value) {
  ImGui::PushID(_parameter_index++);
  ImGui::Text("%s", label_text.data());
  ImGui::SameLine();

  constexpr auto parameter = [](auto text) constexpr {
    return std::string("##") + std::string(text);
  };

  if constexpr (std::is_integral<T>())
    ImGui::SliderInt(parameter(label_text).c_str(), value, min, max);

  else if constexpr (std::is_floating_point<T>())
    ImGui::SliderFloat(parameter(label_text).c_str(), value, min, max);

  ImGui::SameLine();
  if (ImGui::Button("0"))
    *value = default_value;
  gui::enableParameterLearn(value, gui::ParameterType::Float, min, max);
  ImGui::PopID();
}

void Device::draw_imgui_controls() {

  ImGui::PushID(_driver->id().c_str());
  _parameter_index = 0;

  // ImGui::Checkbox("Enable Broadcast", &_enable_broadcast);

  ImGui::Spacing();
  ImGui::Spacing();
  ImGui::Spacing();
  const bool was_paused = paused;
  ImGui::Checkbox("Pause Sensor", &paused);
  if (paused != was_paused)
    _driver->set_paused(paused);
  ImGui::Spacing();
  ImGui::Spacing();
  ImGui::Spacing();

  if (ImGui::Button("Serialize")) serialize_config();
  if (ImGui::Button("Deserialize")) deserialize_config_from_this_device();

  if (paused != was_paused)
    _driver->set_paused(paused);
  ImGui::Spacing();
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
    ImGui::Checkbox(label("x", 0).c_str(), &config.flip_x);
    ImGui::SameLine();
    ImGui::Checkbox(label("y", 0).c_str(), &config.flip_y);
    ImGui::SameLine();
    ImGui::Checkbox(label("z", 0).c_str(), &config.flip_z);

    ImGui::TextDisabled("Crop Input");
    // ImGui can't handle shorts, so we need to use int's then convert
    // back TODO there's probably a better way to do it by defining
    // implicit conversions??
    pc::types::minMax<int> crop_x_in{config.crop_x.min, config.crop_x.max};
    pc::types::minMax<int> crop_y_in{config.crop_y.min, config.crop_y.max};
    pc::types::minMax<int> crop_z_in{config.crop_z.min, config.crop_z.max};
    ImGui::SliderInt2(label("x", 1).c_str(), crop_x_in.arr(), -10000, 10000);
    ImGui::SliderInt2(label("y", 1).c_str(), crop_y_in.arr(), -10000, 10000);
    ImGui::SliderInt2(label("z", 1).c_str(), crop_z_in.arr(), -10000, 10000);
    config.crop_x.min = crop_x_in.min;
    config.crop_x.max = crop_x_in.max;
    config.crop_y.min = crop_y_in.min;
    config.crop_y.max = crop_y_in.max;
    config.crop_z.min = crop_z_in.min;
    config.crop_z.max = crop_z_in.max;

    pc::types::int3 offset_in{config.offset.x, config.offset.y,
                               config.offset.z};
    ImGui::TextDisabled("Offset Output");
    draw_slider<int>("x", &offset_in.x, -10000, 10000);
    draw_slider<int>("y", &offset_in.y, -10000, 10000);
    draw_slider<int>("z", &offset_in.z, -10000, 10000);
    config.offset.x = offset_in.x;
    config.offset.y = offset_in.y;
    config.offset.z = offset_in.z;

    ImGui::TextDisabled("Rotate Output");
    draw_slider<float>("x", &config.rotation_deg.x, -180, 180);
    draw_slider<float>("y", &config.rotation_deg.y, -180, 180);
    draw_slider<float>("z", &config.rotation_deg.z, -180, 180);

    ImGui::TextDisabled("Scale Output");
    draw_slider<float>("uniform", &config.scale, 0.0f, 4.0f);

    ImGui::TextDisabled("Sample");
    draw_slider<int>("s", &config.sample, 1, 10);

    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Device options")) {
    draw_device_controls();
    ImGui::TreePop();
  }

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

} // namespace pc::sensors
