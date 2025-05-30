#include "k4a_device.h"
#include "../../logger.h"
#include "k4a_config.gen.h"
#include <functional>
#include <imgui.h>
#include <unordered_map>
#include <string>

namespace pc::devices {

K4ADevice::K4ADevice(AzureKinectConfiguration &config)
    : DeviceBase<AzureKinectConfiguration>(config) {
  pc::logger->info("Initialising K4ADevice");

  // TODO should be an index of kinect devices
  auto connection_index = pc::devices::attached_devices.size() + 1;

  // TODO just get rid of this 'Driver' concept entirely
  _driver = std::make_unique<K4ADriver>(this->config());

  name = "Azure Kinect " + std::to_string(connection_index);
  if (connection_index == 1) _driver->primary_aligner = true;
  

  // TODO set these parameters from config instead of the reverse
  // auto driver = dynamic_cast<K4ADriver *>(_driver.get());
  // _config.k4a.exposure = driver->get_exposure();
  // _config.k4a.brightness = driver->get_brightness();
  // _config.k4a.contrast = driver->get_contrast();
  // _config.k4a.saturation = driver->get_saturation();
  // _config.k4a.gain = driver->get_gain();
  {
    std::lock_guard lock(K4ADevice::devices_access);
    K4ADevice::attached_devices.push_back(std::ref(*this));
  }
  count++;
}

K4ADevice::~K4ADevice() {
  pc::logger->info("Closing {}", name);
  {
    std::lock_guard lock(K4ADevice::devices_access);
    std::erase_if(K4ADevice::attached_devices,
                  [this](auto &device) { return &(device.get()) == this; });
  }
  count--;
}

DeviceStatus K4ADevice::status() const {
  if (lost_device()) return DeviceStatus::Missing;
  if (!_driver->is_open() || !_driver->is_running())
    return DeviceStatus::Inactive;
  return DeviceStatus::Active;
}

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

  // pc::gui::draw_parameters(_driver->id(),
  //                          parameters::struct_parameters.at(_driver->id()));

  // using pc::gui::slider;
  
  // auto driver = dynamic_cast<K4ADriver *>(_driver.get());

  // if (gui::begin_tree_node("K4A Configuration", _config.k4a.unfolded)) {

  //   static const std::map<k4a_depth_mode_t, std::pair<int, std::string>>
  // 	depth_mode_to_combo_item = {
  // 	    {K4A_DEPTH_MODE_NFOV_2X2BINNED, {0, "NFOV Binned"}},
  // 	    {K4A_DEPTH_MODE_NFOV_UNBINNED, {1, "NFOV Unbinned"}},
  // 	    {K4A_DEPTH_MODE_WFOV_2X2BINNED, {2, "WFOV Binned"}},
  // 	    {K4A_DEPTH_MODE_WFOV_UNBINNED, {3, "WFOV Unbinned"}}};

  //   static const std::string combo_item_string = [] {
  //     std::string items;
  //     for (const auto &[mode, item] : depth_mode_to_combo_item) {
  // 	items += item.second + '\0';
  //     }
  //     return items;
  //   }();

  // 	auto [selected_item_index, label] =
  // 		depth_mode_to_combo_item.at((k4a_depth_mode_t)_config.k4a.depth_mode);

  //   if (ImGui::Combo("Depth Mode", &selected_item_index,
  // 		     combo_item_string.c_str())) {
  //     for (const auto &[mode, item] : depth_mode_to_combo_item) {
  // 	if (item.first == selected_item_index) {
  // 	  _config.k4a.depth_mode = mode;
  // 	  auto driver = dynamic_cast<K4ADriver *>(_driver.get());
  // 	  driver->set_depth_mode(mode);
  // 	  break;
  // 	}
  //     }
  //   }

  //   if (slider(id(), "k4a.exposure", _config.k4a.exposure, 0, 1000000, 10000)) {
  //     driver->set_exposure(_config.k4a.exposure);
  //   }
  //   if (slider(id(), "k4a.brightness", _config.k4a.brightness, 0, 255, 128)) {
  //     driver->set_brightness(_config.k4a.brightness);
  //   }
  //   if (slider(id(), "k4a.contrast", _config.k4a.contrast, 0, 10, 5)) {
  //     driver->set_contrast(_config.k4a.contrast);
  //   }
  //   if (slider(id(), "k4a.saturation", _config.k4a.saturation, 0, 63, 31)) {
  //     driver->set_saturation(_config.k4a.saturation);
  //   }
  //   if (slider(id(), "k4a.gain", _config.k4a.gain, 0, 255, 128)) {
  //     driver->set_gain(_config.k4a.gain);
  //   }

  //   ImGui::TreePop();
  // }

  // if (gui::begin_tree_node("Body tracking", _config.body.unfolded)) {
  //   auto &body = _config.body;
  //   if (ImGui::Checkbox("Enabled", &body.enabled)) {
  //     driver->enable_body_tracking(body.enabled);
  //   }
  //   ImGui::TreePop();
  // }

  // ImGui::Checkbox("Auto tilt", &_config.k4a.auto_tilt.enabled);
  // ImGui::SameLine();
  // if (ImGui::Button("Clear")) driver->clear_auto_tilt();

  // if (_config.k4a.auto_tilt.enabled) {
  //   slider(id(), "k4a.auto_tilt.lerp_factor", _config.k4a.auto_tilt.lerp_factor,
  // 	   0.0f, 1.0f, 0.025f);
  //   slider(id(), "k4a.auto_tilt.threshold", _config.k4a.auto_tilt.threshold,
  // 	   0.0f, 5.0f, 1.0f);
  // }
}

std::string K4ADevice::get_serial_number(const std::size_t device_index) {

  static std::mutex serial_num_query_lock;
  std::unique_lock lock(serial_num_query_lock);
  try {
    auto device = k4a::device::open(device_index);
    auto serial_number = device.get_serialnum();
    device.close();
    return serial_number;
  } catch (k4a::error& e) {
    pc::logger->warn("Unable to get serial number for k4a at index {}",
                     device_index);
    return "";
  }
}

void K4ADevice::reattach(int index) {
  pc::logger->info("Reattaching K4A ({}) at index {}", id(), index);
  auto driver = static_cast<K4ADriver *>(_driver.get());
  driver->reattach(index);
}

} // namespace pc::devices
