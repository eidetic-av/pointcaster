#pragma once

#include "../operators/session_operator_host.h"
#include "../pointer.h"
#include "../uuid.h"
#include "device_config.gen.h"
#include "device_variants.h"
#include <Corrade/Containers/Pointer.h>
#include <imgui.h>
#include <k4abttypes.h>
#include <memory>
#include <mutex>
#include <pointclouds.h>
#include <vector>

namespace pc::devices {

enum class DeviceStatus { Loaded, Active, Inactive, Missing };

// abstract class provides a polymorphic interface over devices, used
// to hold our main collection of attached devices
class IDevice {
public:
  virtual ~IDevice() = default;

  virtual pc::types::PointCloud
  point_cloud(pc::operators::OperatorList operators = {}) = 0;

  virtual DeviceStatus status() const { return DeviceStatus::Loaded; };
  virtual bool draw_controls() { return false; }
};

// all devices live here to be
// accessible throughout the application
inline std::vector<std::shared_ptr<IDevice>> attached_devices;
inline std::mutex devices_access;

// all instances of device configurations that get serialized into our session
// configuration live here
inline std::vector<std::reference_wrapper<DeviceConfigurationVariant>>
    device_configs;
inline std::mutex device_configs_access;

// we forward declare these wrapper functions because we can't include
// parameters.h (for the real funcs) as the serialization code conflicts with
// nvcc
namespace detail {
void declare_device_parameters(DeviceConfigurationVariant &config);
void unbind_device_parameters(const DeviceConfigurationVariant &config);
} // namespace detail

// Extend new devices from DeviceBase and their configurations are automatically
// accessible throughout the app
template <ValidDeviceConfig Config> class DeviceBase : public IDevice {
public:
  explicit DeviceBase(const Config &config) : _config(config) {

    // check if we need to generate an id or if one was passed in with the
    // config (taking into consideration different types of ids)
    if constexpr (std::convertible_to<id_type, std::string_view>) {
      if (this->config().id.empty()) { this->config().id = pc::uuid::word(); }
    } else if constexpr (std::is_integral_v<id_type>) {
      if (std::to_string(this->config().id).empty()) {
        this->config().id = pc::uuid::digit();
      }
    }

    // add the reference to our config to our global list of device
    // configurations
    std::lock_guard lock(device_configs_access);
    device_configs.push_back(std::ref(_config));
    detail::declare_device_parameters(_config);
  }
  DeviceBase(const DeviceBase &other) : _config(other._config) {
    std::lock_guard lock(device_configs_access);
    device_configs.push_back(std::ref(_config));
  }
  DeviceBase(DeviceBase &&other) noexcept : _config(std::move(other._config)) {
    remove_config(&other._config);
    std::lock_guard lock(device_configs_access);
    device_configs.push_back(std::ref(_config));
  }
  DeviceBase &operator=(const DeviceBase &other) {
    if (this != &other) {
      remove_config(&_config);
      _config = other._config;
      std::lock_guard lock(device_configs_access);
      device_configs.push_back(std::ref(_config));
    }
    return *this;
  }
  DeviceBase &operator=(DeviceBase &&other) noexcept {
    if (this != &other) {
      remove_config(&_config);
      remove_config(&other._config);
      _config = std::move(other._config);
      std::lock_guard lock(device_configs_access);
      device_configs.push_back(std::ref(_config));
    }
    return *this;
  }
  virtual ~DeviceBase() { remove_config(&_config); }

  Config &config() { return std::get<Config>(_config); }
  const Config &config() const { return std::get<Config>(_config); }

  using id_type = decltype(std::declval<Config>().id);
  id_type id() const { return config().id; };

private:
  DeviceConfigurationVariant _config;

  static void remove_config(const DeviceConfigurationVariant *ptr) {
    if (!ptr) return;
    std::lock_guard lock(device_configs_access);
    std::erase_if(
        device_configs,
        [ptr](const std::reference_wrapper<DeviceConfigurationVariant> &ref) {
          return &ref.get() == ptr;
        });
    detail::unbind_device_parameters(*ptr);
  }
};

class Device {};
enum class DeviceType { UnknownDevice, K4A, K4W2, Rs2 };

extern pc::types::PointCloud
synthesized_point_cloud(pc::operators::OperatorList &operators);

// TODO make all the k4a stuff more generic
using pc::types::Float4;
using K4ASkeleton =
    std::array<std::pair<pc::types::position, Float4>, K4ABT_JOINT_COUNT>;
extern std::vector<K4ASkeleton> scene_skeletons();

extern pc::types::position global_translate;
extern void draw_global_controls();

} // namespace pc::devices
