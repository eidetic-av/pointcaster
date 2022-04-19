#include "rs2_device.h"
#include "../device.h"
#include "rs2_driver.h"
#include <fmt/format.h>

namespace bob::sensors {

using namespace Magnum;

Rs2Device::Rs2Device() {
  _driver.reset(new Rs2Driver());
  name = fmt::format("rs2 {}", _driver->device_index);
}

std::string Rs2Device::getBroadcastId() {
  return _driver->getId();
}

} // namespace bob::sensors
