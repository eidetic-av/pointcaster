#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

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

struct ObDeviceInfo {
  std::string ip;
  std::string serial_num;
};

class ObContext {
public:
  std::atomic_bool discovering_devices{false};
  std::vector<ObDeviceInfo> discovered_devices{};

  std::mutex start_stop_device_access;

  void discover_devices();
  void discover_devices_async();

  void retain_user();
  void release_user();

  void init_async();

  std::shared_ptr<ob::Context> get_if_ready() const;
  std::shared_ptr<ob::Context> wait_til_ready(int max_wait_seconds = 10) const;
  void run_on_ready(std::function<void()> callback);

  ~ObContext();

private:
  std::atomic<ObContextState> state{ObContextState::Uninitialised};
  std::atomic<std::shared_ptr<ob::Context>> ctx{};
  std::atomic_size_t users{0};

  void shutdown();
};

// global singleton accessor (one per plugin / shared library).
ObContext &orbbec_context();

} // namespace pc::devices
