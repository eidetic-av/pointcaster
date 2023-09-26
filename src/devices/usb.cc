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
  auto *callback_data = static_cast<HotplugCallbackData*>(user_data);

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

    if (device_type == DeviceType::K4A) {
      pc::logger->debug("K4A device plugged in");

      std::string serial_number;
      std::size_t device_index = K4ADevice::active_driver_count();
      try {
        serial_number = K4ADevice::get_serial_number(device_index);
      } catch (k4a::error e) {
	pc::logger->warn("Failed getting serial number of K4A at index {}",
			 device_index);
        pc::logger->warn(e.what());
        return 1;
      }

      DeviceConfiguration config{};

      // load an existing device configuration if it exists
      auto saved_session_devices = callback_data->fetch_session_devices();
      for (const auto &[saved_device_id, existing_config] : saved_session_devices) {
	if (saved_device_id == serial_number) {
	  config = existing_config;
	  break;
	}
      }

      // if an instance for this device already exists (maybe it was unplugged)
      auto reattached = false;
      for (auto &device : Device::attached_devices) {
        if (device->id() == serial_number) {
	  pc::logger->debug("Reattaching existing device");
	  try {
	    device->reattach();
            return 0;
          } catch (k4a::error e) {
	    pc::logger->error("Reattach failed!");
	    pc::logger->error(e.what());
            return 1;
          }
        }
      }

      Device::attached_devices.push_back(std::make_shared<K4ADevice>(config));

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
