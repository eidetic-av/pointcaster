#pragma once

#include "device_plugin.h"
#include "device_variants.h"
#include <Corrade/Containers/Pointer.h>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <pointcaster/point_cloud.h>
#include <profiling/profiling_mutex.h>
#include <string_view>
#include <uuid/uuid.h>
#include <vector>

namespace pc::devices {

enum class DeviceStatus { Loaded, Active, Inactive, Missing };

// all devices live here to be
// accessible throughout the application
inline std::vector<std::shared_ptr<DevicePlugin>> attached_devices;
inline PC_PROFILING_MUTEX(devices_access);

// all instances of device configurations that get serialized into our session
// configuration live here
inline std::vector<std::reference_wrapper<DeviceConfigurationVariant>>
    device_configs;
inline std::optional<std::reference_wrapper<DeviceConfigurationVariant>>
    selected_device_config;
inline PC_PROFILING_MUTEX(device_configs_access);

// we forward declare these wrapper functions because we can't include
// parameters.h (for the real funcs) as the serialization code conflicts with
// nvcc
namespace detail {
void declare_device_parameters(DeviceConfigurationVariant &config);
void unbind_device_parameters(const DeviceConfigurationVariant &config);
} // namespace detail

// Extend new devices from DeviceBase and their configurations are automatically
// accessible throughout the app
template <ValidDeviceConfig Config> class DeviceBase : public DevicePlugin {
public:
  explicit DeviceBase(Corrade::PluginManager::AbstractManager &manager,
                      Corrade::Containers::StringView plugin)
      : DevicePlugin{manager, plugin} {}

  // DeviceBase(const DeviceBase &other) : _config(other._config) {
  //   std::lock_guard lock(device_configs_access);
  //   device_configs.push_back(std::ref(_config));
  // }
  // DeviceBase(DeviceBase &&other) noexcept : _config(std::move(other._config))
  // {
  //   // remove_config(&other._config);
  //   std::lock_guard lock(device_configs_access);
  //   device_configs.push_back(std::ref(_config));
  // }
  // DeviceBase &operator=(const DeviceBase &other) {
  //   if (this != &other) {
  //     // remove_config(&_config);
  //     _config = other._config;
  //     std::lock_guard lock(device_configs_access);
  //     device_configs.push_back(std::ref(_config));
  //   }
  //   return *this;
  // }
  // DeviceBase &operator=(DeviceBase &&other) noexcept {
  //   if (this != &other) {
  //     // remove_config(&_config);
  //     // remove_config(&other._config);
  //     _config = std::move(other._config);
  //     std::lock_guard lock(device_configs_access);
  //     device_configs.push_back(std::ref(_config));
  //   }
  //   return *this;
  // }

  // virtual ~DeviceBase() = default;

  void
  update_config(const DeviceConfigurationVariant &config_variant) override {
    _config = config_variant;
    // check if we need to generate an id or if one was passed in with the
    // config (taking into consideration different types of ids)
    // if constexpr (std::convertible_to<id_type, std::string_view>) {
    //   if (this->config().id.empty()) {
    //     this->config().id = pc::uuid::word();
    //   }
    // } else if constexpr (std::is_integral_v<id_type>) {
    //   if (std::to_string(this->config().id).empty()) {
    //     this->config().id = pc::uuid::digit();
    //   }
    // }

    // TODO: validate where configurations need to be

    // add the reference to our config to our global list of device
    // configurations
    // std::lock_guard lock(device_configs_access);
    // device_configs.push_back(std::ref(_config));
    // detail::declare_device_parameters(_config);
  }

  Config &config() { return std::get<Config>(_config); }
  const Config &config() const { return std::get<Config>(_config); }

  using id_type = decltype(std::declval<Config>().id);
  id_type id() const { return config().id; };

private:
  DeviceConfigurationVariant _config;

  // static void remove_config(const DeviceConfigurationVariant *ptr) {
  //   if (!ptr)
  //     return;
  //   std::lock_guard lock(device_configs_access);
  //   std::erase_if(
  //       device_configs,
  //       [ptr](const std::reference_wrapper<DeviceConfigurationVariant> &ref)
  //       {
  //         return &ref.get() == ptr;
  //       });
  //   // detail::unbind_device_parameters(*ptr);
  // }
};

} // namespace pc::devices
