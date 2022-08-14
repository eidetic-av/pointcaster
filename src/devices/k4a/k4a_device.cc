#include "k4a_device.h"
#include "../device.h"
#include "k4a_driver.h"
#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace bob::sensors {

using namespace Magnum;

namespace k4a {
  static int device_count = 0;
}

K4ADevice::K4ADevice() {
  _driver.reset(new K4ADriver(k4a::device_count++));
  name = fmt::format("k4a {}", _driver->device_index);
}

std::string K4ADevice::getBroadcastId() {
  return _driver->getId();
}

} // namespace bob::sensors
