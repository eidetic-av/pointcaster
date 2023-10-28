#include "k4w2_device.h"
#include "k4w2_driver.h"
#include "../device.h"
#include <fmt/format.h>

namespace pc::devices {

K4W2Device::K4W2Device() {
  _driver.reset(new K4W2Driver());
  name = fmt::format("k4w2 {}", _driver->device_index);
}

std::string K4W2Device::getBroadcastId() {
  return _driver->getId();
}

} // namespace pc::devices
