#include "orbbec_device.h"
#include "orbbec_context.h"

#include <atomic>
#include <cstring>
#include <iostream>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/h/ObTypes.h>
#include <libobsensor/hpp/Context.hpp>
#include <memory>
#include <print>
#include <thread>

namespace pc::devices {

void OrbbecDevice::discover_devices() {
  if (discovering_devices.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  discovered_devices.clear();

  auto ctx = orbbec_context().get_if_ready();
  if (!ctx) {
    std::println("Orbbec context not initialised; skipping discovery");
    discovering_devices.store(false, std::memory_order_release);
    return;
  }

  std::println("Discovering Orbbec devices...");

  try {
    auto device_list = ctx->queryDeviceList();
    const auto count = device_list->deviceCount();

    std::vector<OrbbecDeviceInfo> local_found;
    local_found.reserve(static_cast<std::size_t>(count));

    for (std::size_t i = 0; i < count; ++i) {
      auto device = device_list->getDevice(i);
      auto info = device->getDeviceInfo();
      if (std::strcmp(info->connectionType(), "Ethernet") == 0) {
        local_found.push_back(
            OrbbecDeviceInfo{info->ipAddress(), info->serialNumber()});
      }
    }

    for (const auto &found_device : local_found) {
      std::println("found: {} {}", found_device.ip, found_device.serial_num);
    }

    discovered_devices = std::move(local_found);
  } catch (const ob::Error &e) {
    std::println("Failed to discover Orbbec devices: [{}] {}",
                 e.getName(), e.getMessage());
  } catch (const std::exception &e) {
    std::println("Unknown std::exception during Orbbec device discovery: {}",
                 e.what());
  } catch (...) {
    std::println("Unknown error during Orbbec device discovery");
  }

  std::println("Finished discovering devices.");
  discovering_devices.store(false, std::memory_order_release);
}

OrbbecDevice::OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                           Corrade::Containers::StringView plugin)
    : DevicePlugin(manager, plugin) {
  try {
    orbbec_context().retain_user();
  } catch (...) {
    std::println("Exception during Orbbec context initialisation");
  }

  std::println("Created OrbbecDevice");
}

OrbbecDevice::~OrbbecDevice() {
  orbbec_context().release_user();
  std::println("Destroyed OrbbecDevice");
}

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(OrbbecDevice, pc::devices::OrbbecDevice,
                        "net.pointcaster.DevicePlugin/1.0")
