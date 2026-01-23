#include "orbbec_context.h"

#include <chrono>
#include <concurrentqueue/concurrentqueue.h>
#include <libobsensor/ObSensor.hpp>
#include <mutex>
#include <print>
#include <thread>

namespace {
moodycamel::ConcurrentQueue<std::function<void()>> on_ready_callbacks{};
}

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

  ObContext *self = this;

  std::thread([self] {
    std::shared_ptr<ob::Context> local;

    try {
      local = std::make_shared<ob::Context>();
      local->enableNetDeviceEnumeration(true);
    } catch (const ob::Error &e) {
      std::println("Failed to initialise Orbbec context (ob::Error): [{}] {}",
                   e.getName(), e.getMessage());
      self->state.store(ObContextState::Failed, std::memory_order_release);
      return;
    } catch (const std::exception &e) {
      std::println("Failed to initialise Orbbec context (std::exception): {}",
                   e.what());
      self->state.store(ObContextState::Failed, std::memory_order_release);
      return;
    } catch (...) {
      std::println("Failed to initialise Orbbec context: unknown exception");
      self->state.store(ObContextState::Failed, std::memory_order_release);
      return;
    }

    self->ctx.store(local, std::memory_order_release);
    self->state.store(ObContextState::Ready, std::memory_order_release);

    std::function<void()> cb;
    while (on_ready_callbacks.try_dequeue(cb)) {
      cb();
    }

    std::println("Orbbec context initialised");
  }).detach();
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

void ObContext::discover_devices() {
  if (discovering_devices.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  discovered_devices.clear();

  auto ctx = orbbec_context().get_if_ready();
  if (!ctx) {
    std::println("Orbbec context not initialised; skipping discovery");
    discovering_devices.store(false, std::memory_order_release);
    return;
  }

  std::println("Discovering Orbbec devices...");

  try {
    auto device_list = ctx->queryDeviceList();
    const auto count = device_list->deviceCount();

    std::vector<ObDeviceInfo> local_found;
    local_found.reserve(static_cast<std::size_t>(count));

    for (std::size_t i = 0; i < count; ++i) {
      auto device = device_list->getDevice(i);
      auto info = device->getDeviceInfo();
      if (std::strcmp(info->connectionType(), "Ethernet") == 0) {
        local_found.push_back(
            {info->ipAddress(), info->serialNumber(), info->name()});
      }
    }

    for (const auto &found_device : local_found) {
      std::println("found: {} {} {}", found_device.name, found_device.ip,
                   found_device.serial_num);
    }

    discovered_devices = std::move(local_found);
  } catch (const ob::Error &e) {
    std::println("Failed to discover Orbbec devices: [{}] {}", e.getName(),
                 e.getMessage());
  } catch (const std::exception &e) {
    std::println("Unknown std::exception during Orbbec device discovery: {}",
                 e.what());
  } catch (...) {
    std::println("Unknown error during Orbbec device discovery");
  }

  std::vector<std::function<void()>> post_discovery_callbacks;
  {
    std::lock_guard lock(discovery_callbacks_access);
    post_discovery_callbacks = discovery_callbacks;
  }
  for (auto &cb : post_discovery_callbacks) cb();

  std::println("Finished discovering devices.");
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
    // this was the last user
    shutdown();
  }
}

void ObContext::shutdown() {
  using namespace std::chrono_literals;

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
        local->enableNetDeviceEnumeration(false);
      } catch (...) {
      }
      std::println("Orbbec context destroyed");
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
