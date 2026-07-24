#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ob {
class Context;
class Device;
} // namespace ob

namespace pc::devices {

enum class ObContextState : std::uint8_t {
  Uninitialised,
  Initialising,
  Ready,
  Failed
};

struct ObDeviceInfo {
  std::string ip;
  std::string subnet_mask;
  std::string gateway;

  std::string id; // ob sdk uid
  std::string serial_num;
  std::string name;            // product name as reported by ob sdk
  std::string connection_type; // Ethernet, USB, USB3.2 etc...
  int vendor_id{};
  int product_id{};

  bool is_network() const { return connection_type == "Ethernet"; }
};

enum class ObDeviceModel : std::uint8_t { Unknown, FemtoMega, PulsarSL450 };

constexpr std::array<std::pair<ObDeviceModel, std::string_view>, 2> model_names{
    {{ObDeviceModel::FemtoMega, "Femto Mega"},
     {ObDeviceModel::PulsarSL450, "Pulsar SL450"}}};

ObDeviceModel device_model(const ObDeviceInfo &device_info);

class ObContext {
public:
  std::atomic_bool discovering_devices{false};
  std::vector<ObDeviceInfo> discovered_devices{};
  std::mutex discovered_devices_access;

  std::mutex device_api_access;

  void discover_devices();
  void discover_devices_async();

  void retain_user();
  void release_user();

  void init_async();

  std::shared_ptr<ob::Context> get_if_ready() const;
  std::shared_ptr<ob::Context> wait_til_ready(int max_wait_seconds = 10) const;
  void run_on_ready(std::function<void()> callback);

  void add_discovery_change_callback(std::function<void()> callback);

  void add_to_software_sync_list(std::shared_ptr<ob::Device> device_ptr) {
    std::lock_guard lock(software_sync_device_set_access);
    software_sync_devices.insert(device_ptr);
  }

  void erase_from_software_sync_list(std::shared_ptr<ob::Device> device_ptr) {
    std::lock_guard lock(software_sync_device_set_access);
    software_sync_devices.erase(device_ptr);
  }

  ~ObContext();

private:
  std::atomic<ObContextState> state{ObContextState::Uninitialised};
  std::atomic<std::shared_ptr<ob::Context>> ctx{};
  std::atomic_size_t users{0};

  std::uint64_t device_changed_callback_id{0};

  std::mutex discovery_callbacks_access;
  std::vector<std::function<void()>> discovery_callbacks;

  std::unordered_set<std::shared_ptr<ob::Device>> software_sync_devices;
  std::mutex software_sync_device_set_access;
  std::jthread software_sync_thread{};

  void shutdown();
};

// global singleton accessor (one per plugin / shared library).
ObContext &orbbec_context();

} // namespace pc::devices
