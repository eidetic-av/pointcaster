#pragma once

#include "../pointer.h"
#include "../log.h"
#include "../string_utils.h"
#include "driver.h"
#include <Corrade/Containers/Pointer.h>
#include <imgui.h>
#include "../gui_helpers.h"
#include <mutex>
#include <pointclouds.h>
#include <thread>
#include <vector>
#include <memory>
#include <fstream>
#include <filesystem>
#include <iterator>

namespace bob::sensors {

// using namespace bob::types;

using bob::strings::concat;

enum SensorType { UnknownDevice, K4A, K4W2, Rs2 };

class Device {
public:
  std::string name = "";
  bool is_sensor = true;

  bool pause = false;

  // default values
  // bool flip_x = false;
  // bool flip_y = false;
  // bool flip_z = false;
  // minMax<short> crop_x{-10000, 10000};
  // minMax<short> crop_y{-10000, 10000};
  // minMax<short> crop_z{-10000, 10000};
  // short3 offset = {0, 0, 0};
  // float3 rotation_deg = {-5, 0, 0};
  // float scale = 1.f;
  // int sample = 1;

  // solo freeze
  // bool flip_x = true;
  // bool flip_z = true;
  // minMax<short> crop_z{-10000, 6000};
  // short3 offset = {0, -930, 3000};
  // float scale = 1.2f;

  // looking glass
  // bool flip_x = true;
  // bool flip_z = true;
  // minMax<short> crop_z{-10000, 10000};
  // short3 offset = {0, -930, 1520};
  // float scale = 1.2f;

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

  // TODO why is this public
  std::unique_ptr<Driver> _driver;

  virtual std::string getBroadcastId() = 0;

  bool broadcastEnabled() { return _enable_broadcast; }

  auto pointCloud() {
    return _driver->pointCloud(config);
  };

  uint parameter_index;
  template <typename T>
  void drawSlider(std::string_view label_text, T *value, T min,
		     T max, T default_value = 0) {
    ImGui::PushID(parameter_index++);
    ImGui::Text("%s", label_text.data());
    ImGui::SameLine();

    constexpr auto parameter = [](auto text) constexpr {
      return concat("##", text);
    };

    if constexpr (std::is_integral<T>())
      ImGui::SliderInt(parameter(label_text).c_str(), value, min, max);

    else if constexpr (std::is_floating_point<T>())
      ImGui::SliderFloat(parameter(label_text).c_str(), value, min, max);

    ImGui::SameLine();
    if (ImGui::Button("0")) *value = default_value;
    gui::enableParameterLearn(value, gui::ParameterType::Float, min, max);
    ImGui::PopID();
  }

  void drawImGuiControls() {

    ImGui::PushID(_driver->id().c_str());
    parameter_index = 0;

    // ImGui::Checkbox("Enable Broadcast", &_enable_broadcast);

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    const bool was_paused = pause;
    ImGui::Checkbox("Pause Sensor", &pause);
    if (pause != was_paused) _driver->setPaused(pause);
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();


    constexpr auto output_directory = "data";

    bool serialize = false;
    ImGui::Checkbox("Serialize", &serialize);
    if (serialize) {
      if (!std::filesystem::exists(output_directory))
	std::filesystem::create_directory(output_directory);
      auto [name, data] = serializeConfig();
      std::string filename = concat("data/", name, ".pcc");

      g_log.info("Saving to %s", filename);

      std::ofstream output_file(filename.c_str(), std::ios::out | std::ios::binary);
      output_file.write((const char*)&data[0], data.size());
    }

    bool deserialize = false;
    ImGui::Checkbox("Deserialize", &deserialize);

    if (deserialize) {
      std::string filename = concat("data/", _driver->id(), ".pcc");

      g_log.info("Loading from %s", filename);

      if (std::filesystem::exists(filename)) {

	std::ifstream file(filename, std::ios::binary);

	std::streampos file_size;
	file.seekg(0, std::ios::end);
	file_size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<uint8_t> buffer;
	buffer.reserve(file_size);

	buffer.insert(buffer.begin(),
		      std::istream_iterator<uint8_t>(file),
		      std::istream_iterator<uint8_t>());

        deserializeConfig(buffer);
	g_log.info("Loaded config");

      } else g_log.info("Config doesn't exist");
    }

    if (pause != was_paused) _driver->setPaused(pause);
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

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
        ImGui::Checkbox(label("x", 0).c_str(), &config.flip_x);
        ImGui::SameLine();
        ImGui::Checkbox(label("y", 0).c_str(), &config.flip_y);
        ImGui::SameLine();
        ImGui::Checkbox(label("z", 0).c_str(), &config.flip_z);

        ImGui::TextDisabled("Crop Input");
	// ImGui can't handle shorts, so we need to use int's then convert
	// back TODO there's probably a better way to do it by defining
	// implicit conversions??
	bob::types::minMax<int> crop_x_in{config.crop_x.min, config.crop_x.max};
	bob::types::minMax<int> crop_y_in{config.crop_y.min, config.crop_y.max};
	bob::types::minMax<int> crop_z_in{config.crop_z.min, config.crop_z.max};
        ImGui::SliderInt2(label("x", 1).c_str(), crop_x_in.arr(), -10000, 10000);
        ImGui::SliderInt2(label("y", 1).c_str(), crop_y_in.arr(), -10000, 10000);
        ImGui::SliderInt2(label("z", 1).c_str(), crop_z_in.arr(), -10000, 10000);
	config.crop_x.min = crop_x_in.min;
	config.crop_x.max = crop_x_in.max;
	config.crop_y.min = crop_y_in.min;
	config.crop_y.max = crop_y_in.max;
	config.crop_z.min = crop_z_in.min;
	config.crop_z.max = crop_z_in.max;

	bob::types::int3 offset_in{config.offset.x, config.offset.y, config.offset.z};
        ImGui::TextDisabled("Offset Output");
	drawSlider<int>("x", &offset_in.x, -10000, 10000);
	drawSlider<int>("y", &offset_in.y, -10000, 10000);
	drawSlider<int>("z", &offset_in.z, -10000, 10000);
	config.offset.x = offset_in.x;
	config.offset.y = offset_in.y;
	config.offset.z = offset_in.z;

        ImGui::TextDisabled("Rotate Output");
	drawSlider<float>("x", &config.rotation_deg.x, -180, 180);
	drawSlider<float>("y", &config.rotation_deg.y, -180, 180);
	drawSlider<float>("z", &config.rotation_deg.z, -180, 180);

        ImGui::TextDisabled("Scale Output");
	drawSlider<float>("uniform", &config.scale, 0.0f, 4.0f);

	ImGui::TextDisabled("Sample");
	drawSlider<int>("s", &config.sample, 1, 10);

        ImGui::TreePop()	BeginDisabled();
        Checkbox(concat("##Toggle_Window_", item_name).data(), &window_toggle);
        EndDisabled();;
    }
    if (ImGui::TreeNode("Device options")) {
        drawDeviceSpecificControls();
        ImGui::TreePop();
    }

    ImGui::PopID();
  };

  std::pair<std::string, std::vector<uint8_t>> serializeConfig() const;
  void deserializeConfig(std::vector<uint8_t> data);

protected:
  bool _enable_broadcast = true;

  // implement this to add device-specific options with imgui
  virtual void drawDeviceSpecificControls() {}

  const std::string label(std::string label_text, int index = 0) {
    ImGui::Text("%s", label_text.c_str());
    ImGui::SameLine();
    return "##" + name + "_" + _driver->id() + "_" + label_text + "_" +
	   std::to_string(index);
  }
};

// TODO the following free objects and functions should probably be wrapped
// into some class?
extern std::vector<pointer<Device>> attached_devices;
extern std::mutex devices_access;
extern bob::types::PointCloud synthesizedPointCloud();

extern bob::types::position global_translate;
extern void drawGlobalControls();

} // namespace bob::sensors
