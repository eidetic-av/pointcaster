#include "usb.h"
#include <thread>

namespace bob {

libusb_hotplug_callback_handle _usb_hotplug_callback_handle;
std::vector<std::function<void(bob::sensors::Device *)>> _usb_attach_callbacks;
std::vector<std::function<void(bob::sensors::Device *)>> _usb_detach_callbacks;

std::atomic<bool> run_usb_handler = true;

void initUsb() {
  libusb_init(nullptr);

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

int usbHotplugEvent(struct libusb_context *ctx, struct libusb_device *dev,
			 libusb_hotplug_event event, void *user_data) {
  if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED) {
    for (auto cb : _usb_attach_callbacks) cb(nullptr);
  } else if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT) {
    for (auto cb : _usb_detach_callbacks) cb(nullptr);
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
