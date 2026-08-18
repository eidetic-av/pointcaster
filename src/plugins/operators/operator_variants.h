#pragma once

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

// TODO i think we remove variants entirely for plugins and use runtime
// polymorphism that uses static meta info as much as possible
#include "cluster_extraction/cluster_extraction_config.h"
#include "fringe_removal/fringe_removal_config.h"
#include "range_filter/range_filter_config.h"

namespace pc::operators {

// TODO we absolutely cannot be doing compile time variants for polymorphism,
// when we need the DeviceConfigurations to be able to be dynamically loaded
// from plugins

// we can't garuntee that all variants live inside this codebase
using OperatorConfigurationVariant =
    std::variant<FringeRemovalConfiguration, ClusterExtractionConfiguration,
                 RangeFilterConfiguration>;

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

// the node an operator's config paths sit under...
// its label if it has one and its id otherwise
inline std::string operator_address(const OperatorConfigurationVariant &v) {
  return std::visit(
      [](const auto &config) {
        const auto &label = config.label.value();
        return !label.empty() ? label : config.id;
      },
      v);
}

constexpr bool check_active(const OperatorConfigurationVariant &v) {
  return std::visit([&](auto &config) { return config.active.value(); }, v);
}

} // namespace pc::operators
