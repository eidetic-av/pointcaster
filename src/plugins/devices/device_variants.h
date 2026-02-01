#pragma once

#include "orbbec/orbbec_device_config.h"
#include <concepts>
#include <config/config_variant.h>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace pc::devices {

using DeviceConfigurationVariant = std::variant<OrbbecDeviceConfiguration>;

// compile time utilities

template <typename Callback>
constexpr void for_each_device_config_type(Callback cb) {
  constexpr std::size_t device_config_count =
      std::variant_size_v<DeviceConfigurationVariant>;

  [&]<std::size_t... Indices>(std::index_sequence<Indices...>) {
    (cb.template operator()<
         std::variant_alternative_t<Indices, DeviceConfigurationVariant>>(),
     ...);
  }(std::make_index_sequence<device_config_count>{});
}

} // namespace pc::devices
