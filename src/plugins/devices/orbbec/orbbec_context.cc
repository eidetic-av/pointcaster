#include "orbbec_context.h"

#include <chrono>
#include <libobsensor/ObSensor.hpp>
#include <print>
#include <thread>


namespace pc::devices {

void OrbbecContextSingleton::init_async() {
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

  OrbbecContextSingleton *self = this;

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
    std::println("Orbbec context initialised");
  }).detach();
}

std::shared_ptr<ob::Context> OrbbecContextSingleton::get_if_ready() const {
  if (state.load(std::memory_order_acquire) != ObContextState::Ready) {
    return {};
  }
  return ctx.load(std::memory_order_acquire);
}

void OrbbecContextSingleton::retain_user() {
  users.fetch_add(1, std::memory_order_acq_rel);
  init_async();
}

void OrbbecContextSingleton::release_user() {
  if (users.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    // this was the last user
    shutdown();
  }
}

void OrbbecContextSingleton::shutdown() {
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

OrbbecContextSingleton::~OrbbecContextSingleton() { shutdown(); }

OrbbecContextSingleton &orbbec_context() {
  static OrbbecContextSingleton instance;
  return instance;
}

} // namespace pc::devices
