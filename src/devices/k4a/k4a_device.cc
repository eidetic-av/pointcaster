#include "k4a_device.h"
#include "../../logger.h"
#include <functional>
#include <imgui.h>
#include <unordered_map>
#include <string>

namespace pc::sensors {

K4ADevice::K4ADevice(DeviceConfiguration config) : Device(config) {
  pc::logger->info("Initialising K4ADevice");

  _driver = std::make_unique<K4ADriver>(config);
  if (attached_devices.size() == 0)
    _driver->primary_aligner = true;
  name = "k4a " + std::to_string(_driver->device_index);
  // get any device specific controls needed from the driver
  auto driver = dynamic_cast<K4ADriver *>(_driver.get());
  try {
    _exposure = driver->get_exposure();
    _brightness = driver->get_brightness();
    _contrast = driver->get_contrast();
    _gain = driver->get_gain();
  } catch (::k4a::error &e) {
    pc::logger->error(e.what());
  }

}

K4ADevice::~K4ADevice() { pc::logger->info("Closing {}", name); }

std::string K4ADevice::id() { return _driver->id(); }

void K4ADevice::update_device_control(int *target, int value,
                                      std::function<void(int)> set_func) {
  if (*target != value) {
    try {
      set_func(value);
      *target = value;
    } catch (::k4a::error e) {
      pc::logger->error(e.what());
    }
  }
}

void K4ADevice::draw_device_controls() {

  using pc::gui::draw_slider;
  
  auto driver = dynamic_cast<K4ADriver *>(_driver.get());

  if (gui::begin_tree_node("K4A Configuration", _config.k4a.unfolded)) {

    static const std::map<k4a_depth_mode_t, std::pair<int, std::string>>
	depth_mode_to_combo_item = {
	    {K4A_DEPTH_MODE_NFOV_2X2BINNED, {0, "NFOV Binned"}},
	    {K4A_DEPTH_MODE_NFOV_UNBINNED, {1, "NFOV Unbinned"}},
	    {K4A_DEPTH_MODE_WFOV_2X2BINNED, {2, "WFOV Binned"}},
	    {K4A_DEPTH_MODE_WFOV_UNBINNED, {3, "WFOV Unbinned"}}};

    static const std::string combo_item_string = [] {
      std::string items;
      for (const auto &[mode, item] : depth_mode_to_combo_item) {
	items += item.second + '\0';
      }
      return items;
    }();

    auto [selected_item_index, label] =
        depth_mode_to_combo_item.at(_config.k4a.depth_mode);

    if (ImGui::Combo("Depth Mode", &selected_item_index,
		     combo_item_string.c_str())) {
      for (const auto &[mode, item] : depth_mode_to_combo_item) {
	if (item.first == selected_item_index) {
	  _config.k4a.depth_mode = mode;
	  break;
	}
      }
    }

    ImGui::TreePop();
  }

  if (gui::begin_tree_node("Body tracking", _config.body.unfolded)) {
    auto& body = _config.body;
    auto initially_enabled = body.enabled;
    ImGui::Checkbox("Enabled", &body.enabled);
    if (body.enabled != initially_enabled) driver->enable_body_tracking(body.enabled);
    ImGui::TreePop();
  }

  ImGui::Checkbox("Auto tilt", &_config.k4a.auto_tilt);
  if (_config.k4a.auto_tilt) driver->apply_auto_tilt(true);

  ImGui::SameLine();
  if (ImGui::Button("Clear"))
    driver->apply_auto_tilt(false);

  int exposure = _exposure;
  draw_slider<int>("Exposure (us)", &exposure, 488, 1000000);
  update_device_control(&_exposure, exposure,
                        [&](auto exposure) { driver->set_exposure(exposure); });

  int brightness = _brightness;
  draw_slider<int>("Brightness", &brightness, 0, 255);
  update_device_control(&_brightness, brightness, [&](auto brightness) {
    driver->set_brightness(brightness);
  });

  int contrast = _contrast;
  draw_slider<int>("Contrast", &contrast, 0, 10);
  update_device_control(&_contrast, contrast,
                        [&](auto contrast) { driver->set_contrast(contrast); });

  int saturation = _saturation;
  draw_slider<int>("Saturation", &saturation, 0, 63);
  update_device_control(&_saturation, saturation, [&](auto saturation) {
    driver->set_saturation(saturation);
  });

  int gain = _gain;
  draw_slider<int>("Gain", &gain, 0, 255);
  update_device_control(&_gain, gain,
                        [&](auto gain) { driver->set_gain(gain); });
}

} // namespace pc::sensors
