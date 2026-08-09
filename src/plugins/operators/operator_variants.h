#pragma once

#include <concepts>
#include <type_traits>
#include <utility>
#include <variant>

// TODO i think we remove variants entirely for plugins and use runtime
// polymorphism that uses static meta info as much as possible
#include "cluster_extraction/cluster_extraction_config.h"
#include "fringe_removal/fringe_removal_config.h"

namespace pc::operators {

// TODO we absolutely cannot be doing compile time variants for polymorphism,
// when we need the DeviceConfigurations to be able to be dynamically loaded
// from plugins

// we can't garuntee that all variants live inside this codebase
using OperatorConfigurationVariant =
    std::variant<FringeRemovalConfiguration, ClusterExtractionConfiguration>;

// compile time utilities

template <typename Callback>
constexpr void for_each_operator_config_type(Callback cb) {
  constexpr std::size_t operator_config_count =
      std::variant_size_v<OperatorConfigurationVariant>;

  [&]<std::size_t... Indices>(std::index_sequence<Indices...>) {
    (cb.template operator()<
         std::variant_alternative_t<Indices, OperatorConfigurationVariant>>(),
     ...);
  }(std::make_index_sequence<operator_config_count>{});
}

constexpr auto
operator_info_from_variant(const OperatorConfigurationVariant &v) {
  return std::visit(
      [](const auto &cfg) { return std::make_tuple(cfg.id, cfg.PluginName); },
      v);
}

} // namespace pc::operators
