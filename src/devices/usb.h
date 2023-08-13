#pragma once

#include "device.h"
#include <libusb.h>
#include <map>
#include <optional>
#include <vector>
#include <functional>
#include <thread>

namespace pc {

struct UsbDevice {
  uint8_t libusb_device_address;
  std::shared_ptr<pc::sensors::Device> device;
};

inline libusb_hotplug_callback_handle _usb_hotplug_callback_handle;

// a constant map that holds the type of sensor given by a "product string" that
// is discoverable via libusb. The product string is defined as the vendor id
// followed by the product id, delimited by a colon.
inline const std::map<std::string, pc::sensors::DeviceType>
    UsbDeviceTypeFromProductString = {{"0x045e:0x097d", pc::sensors::K4A},
                                      {"0x045e:0x02c4", pc::sensors::K4W2},
                                      {"0x8086:0x0b64", pc::sensors::Rs2}};

// Given a libusb device descriptor, return the sensor type.
// This is handy since we can get device_descriptors from
// various libusb callbacks.
pc::sensors::DeviceType
getDeviceTypeFromUsbDescriptor(struct libusb_device_descriptor desc);

class UsbMonitor {
public:
  UsbMonitor();
private:
  std::jthread _usb_monitor_thread;
};

std::optional<std::shared_ptr<pc::sensors::Device>>
createUsbDevice(pc::sensors::DeviceType sensor_type);

int usbHotplugEvent(struct libusb_context *ctx, struct libusb_device *dev,
                    libusb_hotplug_event event, void *user_data);

inline std::vector<std::function<void(std::shared_ptr<pc::sensors::Device>)>>
    _usb_attach_callbacks;
void registerUsbAttachCallback(
    std::function<void(std::shared_ptr<pc::sensors::Device>)> cb);

inline std::vector<std::function<void(std::shared_ptr<pc::sensors::Device>)>>
    _usb_detach_callbacks;
void registerUsbDetachCallback(
    std::function<void(std::shared_ptr<pc::sensors::Device>)> cb);

} // namespace pc
