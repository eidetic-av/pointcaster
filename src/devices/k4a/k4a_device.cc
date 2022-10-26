#include "k4a_device.h"
#include <fmt/format.h>

namespace bob::sensors {

namespace k4a {
  static int device_count = 0;
}

K4ADevice::K4ADevice() {
  _driver = std::make_unique<K4ADriver>(k4a::device_count++);
  if (attached_devices.size() == 0) _driver->primary_aligner = true;
  name = fmt::format("k4a {}", _driver->device_index);
}

std::string K4ADevice::getBroadcastId() {
  return _driver->id();
}

} // namespace bob::sensors
