#pragma once

#include "orbbec/orbbec_device_config.h"
#include <variant>

namespace pc::devices {

// helper trait: is T one of the types in Variant
template <typename T, typename Variant> struct is_in_variant;

template <typename T, typename... Ts>
struct is_in_variant<T, std::variant<Ts...>>
    : std::bool_constant<(std::same_as<T, Ts> || ...)> {};

template <typename T, typename Variant>
inline constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;

using DeviceConfigurationVariant = std::variant<OrbbecDeviceConfiguration>;

template <typename T>
concept ValidDeviceConfig = is_in_variant_v<T, DeviceConfigurationVariant>;

} // namespace pc::devices