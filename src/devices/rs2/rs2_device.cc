#include "rs2_device.h"
#include "../../structs.h"
#include "../device.h"
#include "rs2_driver.h"
#include <fmt/format.h>
#include <imgui.h>

namespace pc::sensors {

using namespace Magnum;

Rs2Device::Rs2Device() {
  _driver.reset(new Rs2Driver());
  name = fmt::format("rs2 {}", _driver->device_index);
}

std::string Rs2Device::getBroadcastId() { return _driver->getId(); }

Rs2Driver *driver;
rs2::option_range exposure_range{-1, -1, -1, -1};
float current_exposure;
rs2::option_range gain_range{-1, -1, -1, -1};
float current_gain;
rs2::option_range brightness_range{-1, -1, -1, -1};
float current_brightness;
rs2::option_range contrast_range{-1, -1, -1, -1};
float current_contrast;

void drawColorControl(std::string label, rs2_option option, float *value,
                      rs2::option_range range) {
  if (ImGui::SliderFloat(label.c_str(), value, range.min, range.max)) {
    // make sure we set the exposure to a valid (stepped) value
    *value = range.step * std::round(*value / range.step);
    // and now actually trigger the device option update
    driver->getColorSensor().set_option(option, *value);
  }
}

void Rs2Device::drawDeviceSpecificControls() {
  // TODO can all of the following be encapsulated in a function that works for
  // all settings?

  // check if the boundaries have been set from device values before using them
  if (!driver) {
    driver = dynamic_cast<Rs2Driver *>(_driver.get());
    exposure_range =
        driver->getColorSensor().get_option_range(RS2_OPTION_EXPOSURE);
    gain_range = driver->getColorSensor().get_option_range(RS2_OPTION_GAIN);
    brightness_range =
        driver->getColorSensor().get_option_range(RS2_OPTION_BRIGHTNESS);
    contrast_range =
        driver->getColorSensor().get_option_range(RS2_OPTION_CONTRAST);
  }

  drawColorControl(label("Exposure"), RS2_OPTION_EXPOSURE, &current_exposure,
                   exposure_range);
  drawColorControl(label("Gain"), RS2_OPTION_GAIN, &current_gain, gain_range);
  drawColorControl(label("Brightness"), RS2_OPTION_BRIGHTNESS,
                   &current_brightness, brightness_range);
  drawColorControl(label("Contrast"), RS2_OPTION_CONTRAST, &current_contrast,
                   contrast_range);
}

} // namespace pc::sensors
