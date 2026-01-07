#include "orbbec_device.h"
#include <atomic>
#include <iostream>
#include <libobsensor/h/ObTypes.h>
#include <memory>
#include <print>

namespace {
std::atomic_bool context_initialised{false};
std::atomic<size_t> active_devices{0};

std::unique_ptr<ob::Context> ob_ctx;
// std::unique_ptr<ob::DeviceList> ob_device_list;

// std::mutex start_stop_access;

// std::jthread discovery_thread;

void init_context() {
  // TEMP
  // logger = std::make_unique<TempLogger>();
  // logger->debug("initialising orbbec context");

  // ob_ctx = std::make_unique<ob::Context>();
  // _ob_ctx->enableNetDeviceEnumeration(true);

  // logger->debug("initialised orbbec context");
  // context_initialised = true;
  std::println("eyo");
}

// void destroy_context() {
//   // // logger->debug("destroying orbbec context");
//   // // _ob_ctx->enableNetDeviceEnumeration(false);
//   // // _ob_ctx.reset();
//   // logger->debug("destroyed orbbec context");
// }

} // namespace

  namespace pc::devices {

  // void OrbbecDevice::discover_devices() {
    // discovering_devices = true;
    // discovered_devices.clear();

    // pc::logger->debug("Discovering Orbbec devices...");
    // try {
    //   _ob_device_list = _ob_ctx->queryDeviceList();
    //   auto count = _ob_device_list->deviceCount();

    //   std::vector<OrbbecDeviceInfo> local_found;
    //   local_found.reserve(static_cast<size_t>(count));

    //   for (size_t i = 0; i < count; i++) {
    //     auto device = _ob_device_list->getDevice(i);
    //     auto info = device->getDeviceInfo();
    //     if (std::strcmp(info->connectionType(), "Ethernet") == 0) {
    //       local_found.push_back({info->ipAddress(), info->serialNumber()});
    //     }
    //   }

    //   for(auto& found_device : local_found) {
    //     pc::logger->debug("found: {} {}", found_device.ip,
    //                       found_device.serial_num);
    //   }

    //   discovered_devices = std::move(local_found);

    // } catch (const ob::Error &e) {
    //   pc::logger->error("Failed to discover Orbbec devices: [{}] {}",
    //   e.getName(),
    //                     e.getMessage());
    // } catch (...) {
    //   pc::logger->error("Unknown error during Orbbec device discovery");
    // }

    // pc::logger->debug("Finished discovering devices.");
    // discovering_devices = false;
// }

OrbbecDevice::OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                           Corrade::Containers::StringView plugin)
    : DevicePlugin(manager, plugin) {

      ///// whhhhyyyyyyyyyy


  if (!context_initialised) init_context();
  // discover_devices();
  active_devices++;
  std::println("Created OrbbecDevice");
}

OrbbecDevice::~OrbbecDevice() {
  active_devices--;
  std::println("Destroyed OrbbecDevice");
 }

// DeviceStatus OrbbecDevice::status() const {
//     return DeviceStatus::Inactive;
// }

// pc::types::PointCloud OrbbecDevice::point_cloud() const { return {{}, {}}; }

// void OrbbecDevice::start() { }

// void OrbbecDevice::stop() { }

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(OrbbecDevice, pc::devices::OrbbecDevice,
                        "net.pointcaster.DevicePlugin/1.0")
