#pragma once

#include "orbbec/orbbec_device_config.h"
#include "ply/ply_device_config.h"
#include <concepts>
#include <config/config_variant.h>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace pc::devices {

// TODO we absolutely cannot be doing compile time variants for polymorphism,
// when we need the DeviceConfigurations to be able to be dynamically loaded
// from plugins

// we can't garuntee that all variants live inside this codebase
using DeviceConfigurationVariant =
    std::variant<OrbbecDeviceConfiguration, PlyDeviceConfiguration>;

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

constexpr auto device_info_from_variant(const DeviceConfigurationVariant &v) {
  return std::visit(
      [](const auto &cfg) { return std::make_tuple(cfg.id, cfg.PluginName); },
      v);
}

} // namespace pc::devices
