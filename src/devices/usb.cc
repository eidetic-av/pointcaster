#ifndef WIN32

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
#include <chrono>
#include <memory>

#if WITH_K4A
#include "k4a/k4a_device.h"
#endif
#if WITH_K4W2
#include "k4w2/k4w2_device.h"
#endif
#if WITH_RS2
#include "rs2/rs2_device.h"
#endif

namespace pc::devices {

struct HotplugCallbackData {
  std::function<UsbConfiguration()> fetch_session_config;
  std::function<std::map<std::string, devices::DeviceConfiguration>()>
      fetch_session_devices;
  std::atomic<int> running_callback_count{0};
};

UsbMonitor::UsbMonitor(
    std::function<UsbConfiguration()> fetch_session_config,
    std::function<std::map<std::string, devices::DeviceConfiguration>()>
        fetch_session_devices)

    : _usb_monitor_thread([this, fetch_session_config,
                           fetch_session_devices](auto stop_token) {
	pc::logger->info("Started USB monitor thread");

        libusb_context *libusb_ctx;
        libusb_init(&libusb_ctx);

        libusb_hotplug_callback_handle hotplug_handle;

        auto hotplug_callback_data = std::make_unique<HotplugCallbackData>();
        hotplug_callback_data->fetch_session_config = fetch_session_config;
	hotplug_callback_data->fetch_session_devices = fetch_session_devices;

        libusb_hotplug_register_callback(
            libusb_ctx,
            LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED |
                LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT,
            LIBUSB_HOTPLUG_NO_FLAGS, LIBUSB_HOTPLUG_MATCH_ANY,
            LIBUSB_HOTPLUG_MATCH_ANY, LIBUSB_HOTPLUG_MATCH_ANY,
            UsbMonitor::handle_hotplug_event, hotplug_callback_data.get(),
            &hotplug_handle);

        while (!stop_token.stop_requested()) {
          struct timeval usb_event_timeout = {1, 0}; // 1 second
          int result = libusb_handle_events_timeout_completed(
              libusb_ctx, &usb_event_timeout, nullptr);
          if (result == LIBUSB_ERROR_TIMEOUT)
            continue;
          if (result < 0) {
            pc::logger->warn("libusb event error: {}", result);
          }
        }

	// make sure we don't have any running hotplug event callbacks
        while (hotplug_callback_data->running_callback_count.load() != 0) {
	  std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
	// and clean up
        libusb_hotplug_deregister_callback(libusb_ctx, hotplug_handle);
        libusb_exit(libusb_ctx);

        pc::logger->info("Closed USB monitor thread");
      }) {}

int UsbMonitor::handle_hotplug_event(struct libusb_context *ctx,
				     struct libusb_device *dev,
				     libusb_hotplug_event event,
				     void *user_data) {
  auto *callback_data = static_cast<HotplugCallbackData *>(user_data);

  pc::logger->debug("handle hotplug");

  // keep track of running callbacks...
  // increment the running callback counter
  callback_data->running_callback_count.fetch_add(1);
  auto defer = std::shared_ptr<void>(nullptr, [&](auto) {
    // and decrement when we go out of scope
    callback_data->running_callback_count.fetch_sub(1);
  });

  UsbConfiguration session_config = callback_data->fetch_session_config();
  if (!session_config.open_on_hotplug) return 0;

  struct libusb_device_descriptor desc;
  (void)libusb_get_device_descriptor(dev, &desc);

  auto product_string =
      fmt::format("{:#06x}:{:#06x}", desc.idVendor, desc.idProduct);

  auto it = usb_product_string_to_device_type.find(product_string);
  if (it == usb_product_string_to_device_type.end()) {
    return 0;
  }

  using pc::devices::DeviceType;
  using pc::devices::DeviceConfiguration;
  using pc::devices::K4ADevice;

  auto device_type = it->second;

  if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED) {

    // on a device arriving, get a list of any devices we've previously lost to
    // check if this is them returning

    std::vector<std::shared_ptr<Device>> lost_devices;
    for (const auto &device : Device::attached_devices) {
      if (device->lost_device()) lost_devices.push_back(device);
    }

    if (device_type == DeviceType::K4A) {
      pc::logger->debug("K4A device plugged in");

      static std::mutex serial_k4a_hotplug_handling;
      std::lock_guard<std::mutex> lock(serial_k4a_hotplug_handling);

      // if we have some lost devices to check for, get the k4a's that are
      // waiting to be connected
      std::vector<std::shared_ptr<K4ADevice>> lost_k4a_devices;
      for (const auto &device : lost_devices) {
	auto k4a_device = std::dynamic_pointer_cast<K4ADevice>(device);
	if (k4a_device) lost_k4a_devices.push_back(k4a_device);
      }

      // we need to iterate each connected k4a, make sure we can open a new one
      // succesfully, and look for the serial number to see if it matches any
      // that are currently waiting to be re-plugged

      bool successfully_opened_new_device = false;

      int matching_lost_index = -1;
      std::shared_ptr<K4ADevice> matching_lost_k4a;

      for (int i = 0; i < k4a::device::get_installed_count(); i++) {
        try {
	  // device::open will fail if it's already open, but we still need to
	  // iterate all plugged in k4as because the device indexes change when
	  // new ones are plugged in
	  auto device = k4a::device::open(i);
	  auto serial_number = device.get_serialnum();
          pc::logger->debug("Found new device serial: {}", serial_number);
          device.close();
	  successfully_opened_new_device = true;
	  if (lost_k4a_devices.empty()) break;
	  // check if that serial is one that's currently lost and waiting
          for (const auto &lost_k4a : lost_k4a_devices) {
            if (lost_k4a->id() == serial_number) {
	      matching_lost_index = i;
	      matching_lost_k4a = lost_k4a;
	      break;
            }
          }
        } catch (const k4a::error &e) {
	  pc::logger->debug("Can't retrieve k4a serial number at index {}", i);
        }
	if (matching_lost_index != -1) break;
      }

      // if we weren't able to actually open a device, ignore this hotplug...
      // it means the k4a doesn't have enough power or something like that
      if (!successfully_opened_new_device) {
	pc::logger->warn("Failed to open k4a on USB event (possible power loss)");
        return 0;
      }

      // if it's a completely new device, initialise it fresh
      if (lost_k4a_devices.empty() || matching_lost_index == -1) {
	Device::attached_devices.push_back(
	    std::make_shared<K4ADevice>(DeviceConfiguration{}));
	return 0;
      }

      // or if we found a matching lost device, reattach it!
      if (matching_lost_index != -1) {
	matching_lost_k4a->reattach(matching_lost_index);
	return 0;
      }
    }

  } else if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT) {

    if (device_type == DeviceType::K4A) {

      pc::logger->debug("K4A device unplugged");

    }

  }
  return 0;
}

// pc::devices::DeviceType
// getDeviceTypeFromUsbDescriptor(struct libusb_device_descriptor desc) {
//   // then with info from the descriptor, generate the product string
//   // (ids come from descriptor in hex format)
//   auto product_string =
//       fmt::format("{:#06x}:{:#06x}", desc.idVendor, desc.idProduct);
//   // with the product string we can determine what device type we need to
//   // initialise
//   if (UsbDeviceTypeFromProductString.contains(product_string))
//     return UsbDeviceTypeFromProductString.at(product_string);
//   else
//     return pc::devices::UnknownDevice;
// }

// std::optional<std::shared_ptr<pc::devices::Device>>
// createUsbDevice(pc::devices::DeviceType sensor_type) {

// #if WITH_K4A

//   if (sensor_type == pc::devices::K4A)
//     return std::make_shared<devices::K4ADevice>(
// 	pc::devices::DeviceConfiguration{});

// #endif
// #if WITH_K4W2
//   if (sensor_type == pc::devices::K4W2)
//     return std::make_shared<devices::K4W2Device>();
// #endif
// #if WITH_RS2
//   if (sensor_type == pc::devices::RS2)
//     return std::make_shared<devices::RS2>();
// #endif
//   else
//     return std::nullopt;
// }

// void registerUsbAttachCallback(
//     std::function<void(std::shared_ptr<pc::devices::Device>)> cb) {
//   _usb_attach_callbacks.push_back(cb);
// }

// void registerUsbDetachCallback(
//     std::function<void(std::shared_ptr<pc::devices::Device>)> cb) {
//   _usb_detach_callbacks.push_back(cb);
// }

} // namespace pc

#endif
