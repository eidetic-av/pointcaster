#include "orbbec_context.h"

#include <chrono>
#include <concurrentqueue/moodycamel/concurrentqueue.h>
#include <core/logger/logger.h>
#include <cpplocate/cpplocate.h>
#include <filesystem>
#include <libobsensor/ObSensor.hpp>
#include <mutex>
#include <thread>

namespace {
moodycamel::ConcurrentQueue<std::function<void()>> on_ready_callbacks{};

// the SDK loads a bunch of other libs from an adjacent 'extensions' directory
// (and the sdk location is different on windows and linux)
void configure_extensions_directory() {
  namespace fs = std::filesystem;
  const fs::path exe_dir(cpplocate::getModulePath());
#ifdef _WIN32
  // flat windows layout: plugins/ sits beside the executable
  const fs::path extensions_dir =
      exe_dir / "plugins" / "devices" / "orbbec" / "extensions";
#else
  // linux bin/ layout: plugins/ sits beside the executable's parent dir
  const fs::path extensions_dir =
      exe_dir.parent_path() / "plugins" / "devices" / "orbbec" / "extensions";
#endif
  ob::Context::setExtensionsDirectory(extensions_dir.string().c_str());
}
} // namespace

namespace pc::devices {

void ObContext::init_async() {
  auto existing_state = state.load(std::memory_order_acquire);
  if (existing_state == ObContextState::Ready ||
      existing_state == ObContextState::Initialising) {
    return;
  }

  if (!state.compare_exchange_strong(
          existing_state, ObContextState::Initialising,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    // another thread just started initialisation...
    return;
  }

  pc::logger()->trace("Initialising ObContext");

  ObContext *self = this;

  std::thread([self] {
    std::shared_ptr<ob::Context> local;

    try {
      configure_extensions_directory();
      local = std::make_shared<ob::Context>();
      std::lock_guard api_access(self->device_api_access);
      local->enableNetDeviceEnumeration(true);
      local->enableDeviceClockSync(3600000);
      self->device_changed_callback_id = local->registerDeviceChangedCallback(
          [self](std::shared_ptr<ob::DeviceList>,
                 std::shared_ptr<ob::DeviceList>) {
            pc::logger()->info("Orbbec device change detected ");
            self->discover_devices_async();
          });
    } catch (const ob::Error &e) {
      pc::logger()->error(
          "Failed to initialise Orbbec context (ob::Error in {}): {}",
          e.getFunction(), e.what());
      self->state.store(ObContextState::Failed, std::memory_order_release);
      return;
    } catch (const std::exception &e) {
      pc::logger()->error(
          "Failed to initialise Orbbec context (std::exception): {}", e.what());
      self->state.store(ObContextState::Failed, std::memory_order_release);
      return;
    } catch (...) {
      pc::logger()->error(
          "Failed to initialise Orbbec context: unknown exception");
      self->state.store(ObContextState::Failed, std::memory_order_release);
      return;
    }

    self->ctx.store(local, std::memory_order_release);
    self->state.store(ObContextState::Ready, std::memory_order_release);

    std::function<void()> cb;
    while (on_ready_callbacks.try_dequeue(cb)) {
      cb();
    }

    pc::logger()->trace("Orbbec context initialised");
  }).detach();

  software_sync_thread = std::jthread([this](std::stop_token stop_token) {
    using namespace std::chrono;
    using namespace std::chrono_literals;

    constexpr auto frames_per_second = 5;
    constexpr auto frame_duration = 1'000'000us / frames_per_second;

    auto next_frame_time = steady_clock::now();

    while (!stop_token.stop_requested()) {

      std::vector<std::shared_ptr<ob::Device>> devices_to_trigger;
      {
        std::lock_guard device_set_lock(software_sync_device_set_access);
        for (const auto &device_ptr : software_sync_devices) {
          devices_to_trigger.push_back(device_ptr);
        }
      }
      if (devices_to_trigger.empty()) {
        std::this_thread::sleep_for(250ms);
        next_frame_time = steady_clock::now();
        continue;
      }

      next_frame_time += frame_duration;

      for (const auto &device_ptr : devices_to_trigger) {
        pc::logger()->debug("trigger");
        device_ptr->triggerCapture();
      }

      std::this_thread::sleep_until(next_frame_time);
    }
  });
}

std::shared_ptr<ob::Context> ObContext::get_if_ready() const {
  if (state.load(std::memory_order_acquire) != ObContextState::Ready) {
    return {};
  }
  return ctx.load(std::memory_order_acquire);
}

std::shared_ptr<ob::Context>
ObContext::wait_til_ready(int max_wait_seconds) const {
  using namespace std::chrono_literals;
  constexpr auto sleep_time = 50ms;
  auto max_wait_time = std::chrono::seconds(max_wait_seconds);
  auto wait_time = 0ms;
  std::shared_ptr<ob::Context> ob_ctx;
  while (ob_ctx == nullptr && wait_time < max_wait_time) {
    wait_time += sleep_time;
    std::this_thread::sleep_for(sleep_time);
    ob_ctx = orbbec_context().get_if_ready();
  }
  return ob_ctx;
}

void ObContext::run_on_ready(std::function<void()> callback) {
  auto current_state = state.load(std::memory_order_acquire);
  if (current_state == ObContextState::Ready) {
    callback();
    return;
  }
  on_ready_callbacks.enqueue(std::move(callback));
}

void ObContext::add_discovery_change_callback(std::function<void()> callback) {
  std::scoped_lock lock(discovery_callbacks_access);
  discovery_callbacks.push_back(std::move(callback));
}

ObDeviceModel device_model(const ObDeviceInfo &device_info) {
  for (const auto &[model, name] : model_names) {
    if (device_info.name.contains(name)) return model;
  }
  return ObDeviceModel::Unknown;
}

void ObContext::discover_devices() {
  if (discovering_devices.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  auto ob_ctx = get_if_ready();
  if (!ob_ctx) {
    pc::logger()->trace("Orbbec context not initialised; skipping discovery");
    discovering_devices.store(false, std::memory_order_release);
    return;
  }

  pc::logger()->trace("Discovering Orbbec devices...");

  std::vector<ObDeviceInfo> found_devices;

  try {
    std::lock_guard api_lock(device_api_access);

    auto device_list = ob_ctx->queryDeviceList();
    const auto count = device_list ? device_list->getCount() : 0;

    pc::logger()->trace("Orbbec queryDeviceList found {} devices", count);

    found_devices.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
      try {
        ObDeviceInfo info{.id = device_list->getUid(i),
                          .serial_num = device_list->getSerialNumber(i),
                          .name = device_list->getName(i),
                          .connection_type = device_list->getConnectionType(i),
                          .vendor_id = device_list->getVid(i),
                          .product_id = device_list->getPid(i)};
        if (info.is_network()) {
          info.ip = device_list->getIpAddress(i);
          info.subnet_mask = device_list->getSubnetMask(i);
          info.gateway = device_list->getGateway(i);
        }
        found_devices.push_back(std::move(info));
      } catch (const ob::Error &e) {
        pc::logger()->error(
            "Failed to read info for Orbbec device at index {} ({}): {}", i,
            e.getFunction(), e.what());
      }
    }
  } catch (const ob::Error &e) {
    pc::logger()->error("Failed to discover Orbbec devices ({}): {}",
                        e.getFunction(), e.what());
  } catch (const std::exception &e) {
    pc::logger()->error(
        "Unknown std::exception during Orbbec device discovery: {}", e.what());
  } catch (...) {
    pc::logger()->error("Unknown error during Orbbec device discovery");
  }

  for (const auto &found_device : found_devices) {
    // network devices are identified by ip, usb devices by connection type
    const auto &location = found_device.is_network()
                               ? found_device.ip
                               : found_device.connection_type;
    pc::logger()->info("Discovered '{}' via {} (serial: {}, uid: {}, "
                       "vendor:product 0x{:04x}:0x{:04x})",
                       found_device.name, location, found_device.serial_num,
                       found_device.id, found_device.vendor_id,
                       found_device.product_id);
  }
  if (found_devices.empty()) {
    pc::logger()->info("Orbbec discovery found no devices");
  }

  {
    std::lock_guard list_lock(discovered_devices_access);
    discovered_devices = std::move(found_devices);
  }

  // copy callbacks so they run without holding the lock
  std::vector<std::function<void()>> callbacks;
  {
    std::lock_guard lock(discovery_callbacks_access);
    callbacks = discovery_callbacks;
  }
  for (auto &cb : callbacks) cb();

  pc::logger()->trace("Finished discovering devices.");
  discovering_devices.store(false, std::memory_order_release);
}

void ObContext::discover_devices_async() {
  run_on_ready([self = this] {
    std::thread([self] { self->discover_devices(); }).detach();
  });
}

void ObContext::retain_user() {
  users.fetch_add(1, std::memory_order_acq_rel);
  init_async();
}

void ObContext::release_user() {
  if (users.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    pc::logger()->trace("release ObContext user");
    // this was the last user
    shutdown();
  }
}

void ObContext::shutdown() {
  using namespace std::chrono_literals;

  pc::logger()->trace("Shutting down ObContext...");

  // wait out any in-flight initialisation
  while (state.load(std::memory_order_acquire) ==
         ObContextState::Initialising) {
    std::this_thread::sleep_for(1ms);
  }

  if (state.load(std::memory_order_acquire) == ObContextState::Ready) {
    // take ownership of the shared_ptr and clear the global slot.
    auto local =
        ctx.exchange(std::shared_ptr<ob::Context>{}, std::memory_order_acq_rel);

    if (local) {
      try {
        std::lock_guard api_access(device_api_access);
        if (device_changed_callback_id != 0) {
          local->unregisterDeviceChangedCallback(device_changed_callback_id);
          device_changed_callback_id = 0;
        }
        local->enableNetDeviceEnumeration(false);
      } catch (...) {
      }
      pc::logger()->trace("Orbbec context destroyed");
    }

    {
      std::lock_guard list_lock(discovered_devices_access);
      discovered_devices.clear();
    }

    state.store(ObContextState::Uninitialised, std::memory_order_release);
  }
}

ObContext::~ObContext() {
  shutdown();
}

ObContext &orbbec_context() {
  static ObContext instance;
  return instance;
}

} // namespace pc::devices
