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
    _exposure = driver->getExposure();
    _brightness = driver->getBrightness();
    _contrast = driver->getContrast();
    _gain = driver->getGain();
  } catch (::k4a::error& e) {
    bob::log.error(e.what());
  }
}

K4ADevice::~K4ADevice() {
  bob::log.info("Closing %s", name);
}

std::string K4ADevice::getBroadcastId() {
  return _driver->id();
}

void K4ADevice::updateDeviceControl(int *target, int value,
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

void K4ADevice::drawDeviceSpecificControls() {
  auto driver = dynamic_cast<K4ADriver *>(_driver.get());

  int exposure = _exposure;
  drawSlider<int>("Exposure (us)", &exposure, 488, 1000000);
  updateDeviceControl(&_exposure, exposure, [&](auto exposure) {
    driver->setExposure(exposure);
  });

  int brightness = _brightness;
  drawSlider<int>("Brightness", &brightness, 0, 255);
  updateDeviceControl(&_brightness, brightness, [&](auto brightness) {
    driver->setBrightness(brightness);
  });

  int contrast = _contrast;
  drawSlider<int>("Contrast", &contrast, 0, 10);
  updateDeviceControl(&_contrast, contrast, [&](auto contrast) {
    driver->setContrast(contrast);
  });

  int saturation = _saturation;
  drawSlider<int>("Saturation", &saturation, 0, 63);
  updateDeviceControl(&_saturation, saturation, [&](auto saturation) {
    driver->setSaturation(saturation);
  });

  int gain = _gain;
  drawSlider<int>("Gain", &gain, 0, 255);
  updateDeviceControl(&_gain, gain, [&](auto gain) {
    driver->setGain(gain);
  });
}

} // namespace bob::sensors
