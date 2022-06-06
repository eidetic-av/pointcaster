#pragma once

#include <libusb.h>
#include <vector>
#include <map>
#include <spdlog/spdlog.h>
#include "device.h"
#include "k4a/k4a_device.h"
#include "rs2/rs2_device.h"

namespace bob {

  struct UsbDevice {
    uint8_t libusb_device_address;
    std::shared_ptr<bob::sensors::Device> device;
  };

  extern std::vector<UsbDevice> _current_usb_devices;
  extern libusb_hotplug_callback_handle _usb_hotplug_callback_handle;

  // a constant map that holds the type of sensor given by a "product string" that is
  // discoverable via libusb. The product string is defined as the vendor id followed
  // by the product id, delimited by a colon.
  const std::map<std::string, bob::sensors::SensorType> UsbSensorTypeFromProductString = {
    { "0x045e:0x097c", bob::sensors::K4A }
  };

  // Given a libusb device descriptor, return the sensor type.
  // This is handy since we can get device_descriptors from
  // various libusb callbacks.
  bob::sensors::SensorType getSensorTypeFromUsbDescriptor(struct libusb_device_descriptor desc);

  void initUsb();
  void freeUsb();
  int usbHotplugEvent(struct libusb_context *ctx, struct libusb_device *dev,
		      libusb_hotplug_event event, void* user_data);

  extern std::vector<std::function<void(bob::sensors::Device *)>> _usb_attach_callbacks;
  extern void registerUsbAttachCallback(std::function<void(bob::sensors::Device *)> cb);

  extern std::vector<std::function<void(bob::sensors::Device *)>> _usb_detach_callbacks;
  extern void registerUsbDetachCallback(std::function<void(bob::sensors::Device *)> cb);

}
