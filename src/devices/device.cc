#include "device.h"

namespace bob::sensors {

std::vector<pointer<Device>> attached_devices;
std::mutex devices_access;

PointCloud synthesizedPointCloud() {
  auto result = PointCloud{};
  if (attached_devices.size() == 0) return result;
  std::lock_guard<std::mutex> lock(devices_access);
  for (auto &device : attached_devices)
    result += device->getPointCloud();
  return result;
}

} // namespace bob::sensors
