#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

namespace ob {
class Context;
}

namespace pc::devices {

enum class ObContextState : std::uint8_t {
  Uninitialised,
  Initialising,
  Ready,
  Failed
};

class OrbbecContextSingleton {
public:
  void retain_user();
  void release_user();

  void init_async();

  std::shared_ptr<ob::Context> get_if_ready() const;

  ~OrbbecContextSingleton();

private:
  std::atomic<ObContextState> state{ObContextState::Uninitialised};
  std::atomic<std::shared_ptr<ob::Context>> ctx{};
  std::atomic_size_t users{0};

  void shutdown();
};

// global singleton accessor (one per plugin / shared library).
OrbbecContextSingleton &orbbec_context();

} // namespace pc::devices
