#include "usb.h"
#include "device.h"
#include "libusb.h"
#include <fmt/format.h>
#include "../logger.h"
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

#if WITH_K4A
#include "k4a/k4a_device.h"
#endif
#if WITH_K4W2
#include "k4w2/k4w2_device.h"
#endif
#if WITH_RS2
#include "rs2/rs2_device.h"
#endif

namespace pc {

UsbMonitor::UsbMonitor() {
  libusb_init(nullptr);

  // std::lock_guard<std::mutex> lock(pc::devices::Device::devices_access);
  // // get attached USB devices and check if any are sensors we have
  // // drivers for

  // struct libusb_device **device_list;
  // int device_count = libusb_get_device_list(nullptr, &device_list);
  // for (int i = 0; i < device_count; i++) {
  //   struct libusb_device *device = device_list[i];
  //   struct libusb_device_descriptor desc;
  //   libusb_get_device_descriptor(device, &desc);
  //   auto sensor_type = getDeviceTypeFromUsbDescriptor(desc);
  //   auto attached_device = createUsbDevice(sensor_type);
  //   // pc::devices::attached_devices.push_back(std::move(attached_device));
  //   // if (attached_device != nullptr) {
  //   //   for (auto cb : _usb_attach_callbacks) cb(attached_device);
  //   // }
  // }
  // libusb_free_device_list(device_list, 1);

  // // if we were able to initialise a new device, run any attach event
  // // callbacks
  // for (auto cb : _usb_attach_callbacks) cb(attached_device);

  // register a callback for hotplug access
  auto libusb_rc_result = libusb_hotplug_register_callback(
      nullptr,
      LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED | LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT, 0,
      LIBUSB_HOTPLUG_MATCH_ANY, LIBUSB_HOTPLUG_MATCH_ANY,
      LIBUSB_HOTPLUG_MATCH_ANY, usbHotplugEvent, nullptr,
      &_usb_hotplug_callback_handle);

  if (libusb_rc_result != LIBUSB_SUCCESS) {
    pc::logger->error("USB initialisation error {}", libusb_rc_result);
    return;
  }

  // start a thread that waits for libusb events and runs them
  _usb_monitor_thread = std::jthread([&](auto stop_token) {
    pc::logger->info("Started USB monitor thread");
    while (!stop_token.stop_requested()) {
      struct timeval usb_event_timeout = {1, 0}; // 1 second
      int result = libusb_handle_events_timeout_completed(
          nullptr, &usb_event_timeout, nullptr);
      if (result == LIBUSB_ERROR_TIMEOUT)
        continue;
      if (result < 0) {
        pc::logger->warn("libusb event error: {}", result);
      }
    }
    // when run_usb_handler is false, finalise the thread
    // by freeing all libusb resources
    libusb_hotplug_deregister_callback(nullptr, _usb_hotplug_callback_handle);
    libusb_exit(nullptr);
    pc::logger->info("Closed USB monitor thread");
  });
}

pc::devices::DeviceType
getDeviceTypeFromUsbDescriptor(struct libusb_device_descriptor desc) {
  // then with info from the descriptor, generate the product string
  // (ids come from descriptor in hex format)
  auto product_string =
      fmt::format("{:#06x}:{:#06x}", desc.idVendor, desc.idProduct);
  // with the product string we can determine what device type we need to
  // initialise
  if (UsbDeviceTypeFromProductString.contains(product_string))
    return UsbDeviceTypeFromProductString.at(product_string);
  else
    return pc::devices::UnknownDevice;
}

std::optional<std::shared_ptr<pc::devices::Device>>
createUsbDevice(pc::devices::DeviceType sensor_type) {

#if WITH_K4A

  if (sensor_type == pc::devices::K4A)
    return std::make_shared<devices::K4ADevice>(
	pc::devices::DeviceConfiguration{});

#endif
#if WITH_K4W2
  if (sensor_type == pc::devices::K4W2)
    return std::make_shared<devices::K4W2Device>();
#endif
#if WITH_RS2
  if (sensor_type == pc::devices::RS2)
    return std::make_shared<devices::RS2>();
#endif
  else
    return std::nullopt;
}

int usbHotplugEvent(struct libusb_context *ctx, struct libusb_device *dev,
                    libusb_hotplug_event event, void *user_data) {
  if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED) {
    // when a device is plugged in, get the device descriptor

    struct libusb_device_descriptor desc;
    (void)libusb_get_device_descriptor(dev, &desc);
    auto sensor_type = getDeviceTypeFromUsbDescriptor(desc);
    auto attached_device = createUsbDevice(sensor_type);
    if (!attached_device)
      return 1;
    pc::devices::Device::attached_devices.push_back(attached_device.value());

    // if we were able to initialise a new device, run any attach event
    // callbacks
    for (auto cb : _usb_attach_callbacks)
      cb(attached_device.value());

  } else if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT) {

    // TODO handle device detach
    for (auto cb : _usb_detach_callbacks)
      cb(nullptr);
  }
  return 0;
}

void registerUsbAttachCallback(
    std::function<void(std::shared_ptr<pc::devices::Device>)> cb) {
  _usb_attach_callbacks.push_back(cb);
}

void registerUsbDetachCallback(
    std::function<void(std::shared_ptr<pc::devices::Device>)> cb) {
  _usb_detach_callbacks.push_back(cb);
}

} // namespace pc
