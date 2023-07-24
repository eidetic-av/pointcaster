#include "k4a_device.h"
#include "../../log.h"
#include "../../string_utils.h"
#include <imgui.h>
#include <functional>

namespace bob::sensors {

using bob::strings::concat;

K4ADevice::K4ADevice() {
  bob::log.info("Initialising K4ADevice");

  _driver = std::make_unique<K4ADriver>();
  if (attached_devices.size() == 0) _driver->primary_aligner = true;
  name = concat("k4a ", std::to_string(_driver->device_index));
  // get any device specific controls needed from the driver
  auto driver = dynamic_cast<K4ADriver*>(_driver.get());
  try {
    _exposure = driver->get_exposure();
    _brightness = driver->get_brightness();
    _contrast = driver->get_contrast();
    _gain = driver->get_gain();
  } catch (::k4a::error& e) {
    bob::log.error(e.what());
  }

  // load last saved configuration on startup
  deserialize_config_from_this_device();
}

K4ADevice::~K4ADevice() {
  bob::log.info("Closing %s", name);
}

std::string K4ADevice::get_broadcast_id() {
  return _driver->id();
}

void K4ADevice::update_device_control(int *target, int value,
                      std::function<void(int)> set_func){
  if (*target != value) {
    try {
      set_func(value);
      *target = value;
    } catch (::k4a::error e) {
      bob::log.error(e.what());
    }
  }
}

void K4ADevice::draw_device_controls() {
  auto driver = dynamic_cast<K4ADriver *>(_driver.get());

  int exposure = _exposure;
  draw_slider<int>("Exposure (us)", &exposure, 488, 1000000);
  update_device_control(&_exposure, exposure, [&](auto exposure) {
    driver->set_exposure(exposure);
  });

  int brightness = _brightness;
  draw_slider<int>("Brightness", &brightness, 0, 255);
  update_device_control(&_brightness, brightness, [&](auto brightness) {
    driver->set_brightness(brightness);
  });

  int contrast = _contrast;
  draw_slider<int>("Contrast", &contrast, 0, 10);
  update_device_control(&_contrast, contrast, [&](auto contrast) {
    driver->set_contrast(contrast);
  });

  int saturation = _saturation;
  draw_slider<int>("Saturation", &saturation, 0, 63);
  update_device_control(&_saturation, saturation, [&](auto saturation) {
    driver->set_saturation(saturation);
  });

  int gain = _gain;
  draw_slider<int>("Gain", &gain, 0, 255);
  update_device_control(&_gain, gain, [&](auto gain) {
    driver->set_gain(gain);
  });
}

} // namespace bob::sensors
