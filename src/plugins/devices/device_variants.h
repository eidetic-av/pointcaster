#pragma once

#include "orbbec/orbbec_device_config.h"
#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace pc::devices {

using DeviceConfigurationVariant = std::variant<OrbbecDeviceConfiguration>;

// helper trait: is T one of the types in Variant
template <typename T, typename Variant> struct is_in_variant;

template <typename T, typename... Ts>
struct is_in_variant<T, std::variant<Ts...>>
    : std::bool_constant<(std::same_as<T, Ts> || ...)> {};

template <typename T, typename Variant>
inline constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;

template <typename T>
concept ValidDeviceConfig = is_in_variant_v<T, DeviceConfigurationVariant>;

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
