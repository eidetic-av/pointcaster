#include "usb.h"
#include "libusb.h"
#include <iomanip>
#include <sstream>
#include <thread>
#include "device.h"

#if WITH_K4A
#include "k4a/k4a_device.h"
#endif
#if WITH_K4W2
#include "k4w2/k4w2_device.h"
#endif
#if WITH_RS2
#include "rs2/rs2_device.h"
#endif

namespace bob {

libusb_hotplug_callback_handle _usb_hotplug_callback_handle;
std::vector<std::function<void(bob::sensors::Device *)>> _usb_attach_callbacks;
std::vector<std::function<void(bob::sensors::Device *)>> _usb_detach_callbacks;

std::atomic<bool> run_usb_handler = true;

void initUsb() {
  libusb_init(nullptr);

  // get attached USB devices and check if any are sensors we have
  // drivers for
  struct libusb_device **device_list;
  int device_count = libusb_get_device_list(nullptr, &device_list);
  for (int i = 0; i < device_count; i++) {
    struct libusb_device *device = device_list[i];
    struct libusb_device_descriptor desc;
    libusb_get_device_descriptor(device, &desc);
    auto sensor_type = getSensorTypeFromUsbDescriptor(desc);
    auto attached_device = createUsbDevice(sensor_type);
    if (attached_device != nullptr) {
      for (auto cb : _usb_attach_callbacks) cb(attached_device);
    }
  }
  libusb_free_device_list(device_list, 1);

  // // if we were able to initialise a new device, run any attach event callbacks
  // for (auto cb : _usb_attach_callbacks)
  //   cb(attached_device);

  // register a callback for hotplug access
  auto libusb_rc_result = libusb_hotplug_register_callback(
      nullptr,
      LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED | LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT, 0,
      LIBUSB_HOTPLUG_MATCH_ANY, LIBUSB_HOTPLUG_MATCH_ANY,
      LIBUSB_HOTPLUG_MATCH_ANY, usbHotplugEvent, nullptr,
      &_usb_hotplug_callback_handle);

  if (libusb_rc_result != LIBUSB_SUCCESS) {
    spdlog::error("USB initialisation error {}", libusb_rc_result);
    return;
  }

  // start a thread that waits for libusb events and runs them
  std::thread usb_handler_thread([&]() {
    spdlog::info("Initialised libusb");
    while (run_usb_handler) {
      // the following is blocking, so no need for thread sleep
      libusb_handle_events_completed(nullptr, nullptr);
    }
    // when run_usb_handler is false, finalise the thread
    // by freeing all libusb resources
    libusb_hotplug_deregister_callback(nullptr, _usb_hotplug_callback_handle);
    libusb_exit(nullptr);
    spdlog::info("Freed libusb resources");
  });
  usb_handler_thread.detach();
}

void freeUsb() {
  run_usb_handler = false;
}

bob::sensors::SensorType
getSensorTypeFromUsbDescriptor(struct libusb_device_descriptor desc) {
    // then with info from the descriptor, generate the product string...
    // although we need the ids as hex strings
    std::stringstream hex_stream;
    hex_stream << "0x" << std::setfill('0') << std::setw(sizeof(uint16_t) * 2)
		     << std::hex << desc.idVendor;
    auto vendor_id = hex_stream.str();
    // clear the stream and get the product id
    hex_stream.str("");
    hex_stream << "0x" << std::setfill('0') << std::setw(sizeof(uint16_t) * 2)
		     << std::hex << desc.idProduct;
    auto product_id = hex_stream.str();

    auto product_string = vendor_id + ":" + product_id;

    // with the product string we can determine what device type we need to initialise
    if (UsbSensorTypeFromProductString.contains(product_string))
	    return UsbSensorTypeFromProductString.at(product_string);
    else return bob::sensors::UnknownDevice;
}

bob::sensors::Device* createUsbDevice(bob::sensors::SensorType sensor_type) {
    if (sensor_type == bob::sensors::K4A) return new sensors::K4ADevice();
    if (sensor_type == bob::sensors::K4W2) return new sensors::K4W2Device();
    if (sensor_type == bob::sensors::Rs2) return new sensors::Rs2Device();
    return nullptr;
}

int usbHotplugEvent(struct libusb_context *ctx, struct libusb_device *dev,
			 libusb_hotplug_event event, void *user_data) {
  if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED) {
    // when a device is plugged in, get the device descriptor
    struct libusb_device_descriptor desc;
    (void) libusb_get_device_descriptor(dev, &desc);
    auto sensor_type = getSensorTypeFromUsbDescriptor(desc);

    auto attached_device = createUsbDevice(sensor_type);
    if (attached_device == nullptr) return 1;

    // if we were able to initialise a new device, run any attach event callbacks
    for (auto cb : _usb_attach_callbacks) cb(attached_device);

  } else if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT) {

    // TODO handle device detach
    // for (auto cb : _usb_detach_callbacks) cb(nullptr);

  }
  return 0;
}

void registerUsbAttachCallback(std::function<void(bob::sensors::Device *)> cb) {
  _usb_attach_callbacks.push_back(cb);
}

void registerUsbDetachCallback(std::function<void(bob::sensors::Device *)> cb) {
  _usb_detach_callbacks.push_back(cb);
}

} // namespace bob
