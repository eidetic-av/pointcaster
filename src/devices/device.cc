#include "device.h"
#include <spdlog/spdlog.h>

#ifndef __CUDACC__
#include <zpp_bits.h>
#endif

namespace bob::sensors {

std::vector<pointer<Device>> attached_devices;
std::mutex devices_access;

bob::types::PointCloud synthesizedPointCloud() {
  auto result = bob::types::PointCloud{};
  if (attached_devices.size() == 0)
    return result;
  // std::lock_guard<std::mutex> lock(devices_access);
  for (auto &device : attached_devices)
    result += device->pointCloud();
  return result;
}

  std::pair<std::string, std::vector<uint8_t>> Device::serializeConfig() const {
  spdlog::debug("Serializing device configuration for '{}'", this->name);
  std::vector<uint8_t> data;
  auto out = zpp::bits::out(data);
  auto success = out(config);
  return {this->_driver->id(), data};
};

void
Device::deserializeConfig(std::vector<uint8_t> buffer) {
  auto in = zpp::bits::in(buffer);
  auto success = in(this->config);
}

} // namespace bob::sensors
