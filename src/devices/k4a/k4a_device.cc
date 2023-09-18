#include "k4a_device.h"
#include "../../logger.h"
#include <functional>
#include <imgui.h>
#include <unordered_map>
#include <string>

namespace pc::devices {

K4ADevice::K4ADevice(DeviceConfiguration config) : Device(config) {
  pc::logger->info("Initialising K4ADevice");

  _driver = std::make_unique<K4ADriver>(config);
  if (attached_devices.size() == 0)
    _driver->primary_aligner = true;
  name = "Azure Kinect " + std::to_string(_driver->device_index + 1);

  // TODO set these parameters from config instead of the reverse
  auto driver = dynamic_cast<K4ADriver *>(_driver.get());
  _config.k4a.exposure = driver->get_exposure();
  _config.k4a.brightness = driver->get_brightness();
  _config.k4a.contrast = driver->get_contrast();
  _config.k4a.saturation = driver->get_saturation();
  _config.k4a.gain = driver->get_gain();

  // declare_parameters(_driver->id(), _config);
}

K4ADevice::~K4ADevice() {
  pc::logger->info("Closing {}", name);
}

std::string K4ADevice::id() { return _driver->id(); }

void K4ADevice::update_device_control(int *target, int value,
                                      std::function<void(int)> set_func) {
  if (*target != value) {
    try {
      set_func(value);
      *target = value;
    } catch (k4a::error e) {
      pc::logger->error(e.what());
    }
  }
}

void K4ADevice::draw_device_controls() {

  using pc::gui::slider;
  
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
	  auto driver = dynamic_cast<K4ADriver *>(_driver.get());
	  driver->set_depth_mode(mode);
	  break;
	}
      }
    }

    if (slider(id(), "k4a.exposure", _config.k4a.exposure, 0, 1000000, 10000)) {
      driver->set_exposure(_config.k4a.exposure);
    }
    if (slider(id(), "k4a.brightness", _config.k4a.brightness, 0, 255, 128)) {
      driver->set_brightness(_config.k4a.brightness);
    }
    if (slider(id(), "k4a.contrast", _config.k4a.contrast, 0, 10, 5)) {
      driver->set_contrast(_config.k4a.contrast);
    }
    if (slider(id(), "k4a.saturation", _config.k4a.saturation, 0, 63, 31)) {
      driver->set_saturation(_config.k4a.saturation);
    }
    if (slider(id(), "k4a.gain", _config.k4a.gain, 0, 255, 128)) {
      driver->set_gain(_config.k4a.gain);
    }

    ImGui::TreePop();
  }

  if (gui::begin_tree_node("Body tracking", _config.body.unfolded)) {
    auto &body = _config.body;
    auto initially_enabled = body.enabled;
    ImGui::Checkbox("Enabled", &body.enabled);
    if (body.enabled != initially_enabled)
      driver->enable_body_tracking(body.enabled);
    ImGui::TreePop();
  }

  ImGui::Checkbox("Auto tilt", &_config.k4a.auto_tilt);
  if (_config.k4a.auto_tilt) driver->apply_auto_tilt(true);

  ImGui::SameLine();
  if (ImGui::Button("Clear"))
    driver->apply_auto_tilt(false);
}

} // namespace pc::devices
