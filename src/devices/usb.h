#pragma once

#ifndef WIN32

#include "device.h"
#include <functional>
#include <libusb.h>
#include <map>
#include <optional>
#include <thread>
#include <vector>
#include "usb_config.gen.h"

namespace pc::devices {

struct UsbDevice {
  uint8_t libusb_device_address;
  std::shared_ptr<pc::devices::Device> device;
};

// a constant map that holds the type of sensor given by a "product string" that
// is discoverable via libusb. The product string is defined as the vendor id
// followed by the product id, delimited by a colon.
inline const std::unordered_map<std::string_view, pc::devices::DeviceType>
    usb_product_string_to_device_type = {
        {"0x045e:0x097d", pc::devices::DeviceType::K4A},
        {"0x8086:0x0b64", pc::devices::DeviceType::Rs2}};

// Given a libusb device descriptor, return the sensor type.
// This is handy since we can get device_descriptors from
// various libusb callbacks.
pc::devices::DeviceType
getDeviceTypeFromUsbDescriptor(struct libusb_device_descriptor desc);

class UsbMonitor {
public:
  UsbMonitor(std::function<UsbConfiguration()> fetch_session_config,
	     std::function<std::map<std::string, devices::DeviceConfiguration>()>
		 fetch_session_devices);

private:
  std::jthread _usb_monitor_thread;

  static int handle_hotplug_event(struct libusb_context *ctx,
                                  struct libusb_device *dev,
                                  libusb_hotplug_event event, void *user_data);
};

} // namespace pc::devices

#endif
