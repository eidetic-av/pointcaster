#include "orbbec_device.h"
#include "orbbec_context.h"

#include <atomic>
#include <cstring>
#include <iostream>
#include <chrono>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/h/ObTypes.h>
#include <libobsensor/hpp/Context.hpp>
#include <memory>
#include <print>
#include <thread>

namespace pc::devices {


OrbbecDevice::OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                           Corrade::Containers::StringView plugin)
    : DevicePlugin(manager, plugin) {
  try {
    orbbec_context().retain_user();
    orbbec_context().discover_devices_async();
  } catch (...) {
    std::println("Exception during Orbbec context initialisation");
  }

  // for testing status changed
  std::thread([&](){
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(10s);
    notify_status_changed(status());
  }).detach();

  std::println("Created OrbbecDevice");
}

OrbbecDevice::~OrbbecDevice() {
  orbbec_context().release_user();
  std::println("Destroyed OrbbecDevice");
}

DeviceStatus OrbbecDevice::status() const {
  auto ctx = orbbec_context().get_if_ready();
  if (!ctx) return DeviceStatus::Unloaded;
  return DeviceStatus::Loaded;
};

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(OrbbecDevice, pc::devices::OrbbecDevice,
                        "net.pointcaster.DevicePlugin/1.0")
