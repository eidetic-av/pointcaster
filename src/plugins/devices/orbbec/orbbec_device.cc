#include "orbbec_device.h"
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <libobsensor/h/ObTypes.h>
#include <memory>
#include <mutex>
#include <print>
#include <thread>

namespace {

// Counts *all* users of ob_ctx: devices + in-flight discovery calls
std::atomic_size_t context_users{0};

std::unique_ptr<ob::Context> ob_ctx;
std::mutex ob_ctx_access;
std::condition_variable ob_ctx_cv;

enum class ObContextState { Uninitialised, Initialising, Ready, Destroying };
std::atomic<ObContextState> ctx_state = ObContextState::Uninitialised;

void destroy_context(); // forward

inline void retain_context() {
  context_users.fetch_add(1, std::memory_order_acq_rel);
}

inline void release_context() {
  const auto previous =
      context_users.fetch_sub(1, std::memory_order_acq_rel);
  if (previous == 1) {
    destroy_context();
  }
}

void init_context() {
  auto existing_state = ctx_state.load();
  if (existing_state == ObContextState::Ready ||
      existing_state == ObContextState::Initialising ||
      existing_state == ObContextState::Destroying) {
    return;
  }

  std::unique_lock<std::mutex> lock{ob_ctx_access};

  ctx_state = ObContextState::Initialising;

  std::thread([] {
    std::unique_ptr<ob::Context> ctx;
    try {
      ctx = std::make_unique<ob::Context>();
      ctx->enableNetDeviceEnumeration(true);
    } catch (...) {
      std::lock_guard<std::mutex> lk{ob_ctx_access};
      ctx_state = ObContextState::Uninitialised;
      ob_ctx_cv.notify_all();
      std::println("Failed to initialise Orbbec context");
      return;
    }

    {
      std::lock_guard<std::mutex> lk{ob_ctx_access};
      ob_ctx = std::move(ctx);
      ctx_state = ObContextState::Ready;
    }

    ob_ctx_cv.notify_all();
  }).detach();
}

void destroy_context() {
  std::unique_lock<std::mutex> lock{ob_ctx_access};

  if (ctx_state == ObContextState::Uninitialised) return;

  if (ctx_state == ObContextState::Initialising) {
    ob_ctx_cv.wait(lock,
                   [] { return ctx_state != ObContextState::Initialising; });
  }

  if (ctx_state == ObContextState::Ready) {
    ctx_state = ObContextState::Destroying;

    auto ctx = std::move(ob_ctx);
    lock.unlock();

    try {
      ctx->enableNetDeviceEnumeration(false);
    } catch (...) {
    }

    ctx.reset();

    lock.lock();
    ctx_state = ObContextState::Uninitialised;
    ob_ctx_cv.notify_all();
  }
}

} // namespace

namespace pc::devices {

void OrbbecDevice::discover_devices() {
  if (discovering_devices.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  // discovery itself counts as a user of the context
  retain_context();

  discovered_devices.clear();

  ob::Context *ctx = nullptr;
  {
    std::lock_guard<std::mutex> lock{ob_ctx_access};
    ctx = ob_ctx.get();
  }

  if (!ctx) {
    std::println("Orbbec context not initialised; skipping discovery");
    discovering_devices.store(false, std::memory_order_release);
    release_context();
    return;
  }

  std::println("Discovering Orbbec devices...");
  try {
    auto device_list = ctx->queryDeviceList();
    auto count = device_list->deviceCount();

    std::vector<OrbbecDeviceInfo> local_found;
    local_found.reserve(static_cast<size_t>(count));

    for (size_t i = 0; i < count; i++) {
      auto device = device_list->getDevice(i);
      auto info = device->getDeviceInfo();
      if (std::strcmp(info->connectionType(), "Ethernet") == 0) {
        local_found.push_back({info->ipAddress(), info->serialNumber()});
      }
    }

    for (auto &found_device : local_found) {
      std::println("found: {} {}", found_device.ip, found_device.serial_num);
    }

    discovered_devices = std::move(local_found);

  } catch (const ob::Error &e) {
    std::println("Failed to discover Orbbec devices: [{}] {}",
                 e.getName(), e.getMessage());
  } catch (...) {
    std::println("Unknown error during Orbbec device discovery");
  }

  std::println("Finished discovering devices.");
  discovering_devices.store(false, std::memory_order_release);

  release_context();
}

OrbbecDevice::OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                           Corrade::Containers::StringView plugin)
    : DevicePlugin(manager, plugin) {

  init_context();
  retain_context();
  std::thread([]() {
    using namespace std::chrono_literals;
    constexpr auto max_wait_time = 10s;
    constexpr auto sleep_time = 20ms;
    auto wait_time = 0ms;
    while (ctx_state.load() != ObContextState::Ready &&
           wait_time < max_wait_time) {
      std::this_thread::sleep_for(sleep_time);
      wait_time += sleep_time;
    }
    if (wait_time > max_wait_time) return;
    OrbbecDevice::discover_devices();
  }).detach();
  std::println("Created OrbbecDevice");
}

OrbbecDevice::~OrbbecDevice() {
  release_context();
  std::println("Destroyed OrbbecDevice");
}

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(OrbbecDevice, pc::devices::OrbbecDevice,
                        "net.pointcaster.DevicePlugin/1.0")
