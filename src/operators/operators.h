#pragma once

#include "../serialization.h"
#include "../structs.h"
#include "noise_operator.h"
#include "knn_filter_operator.h"
#include "operator_types.h"
#include <variant>
#include <functional>

namespace pc::operators {

using OperatorConfigurationVariant = std::variant<
  NoiseOperatorConfiguration,
  KNNFilterOperatorConfiguration
>;

// Extract types from variant into a tuple
template<typename Variant>
struct VariantTypes;

template<typename... Args>
struct VariantTypes<std::variant<Args...>> {
    using type = std::tuple<Args...>;
};

template <size_t Index = 0, typename TupleType, typename Func>
constexpr void apply_to_variant_types(TupleType, Func func) {
    if constexpr (Index < std::tuple_size_v<TupleType>) {
      using CurrentType = std::tuple_element_t<Index, TupleType>;
      func(CurrentType{});

      apply_to_variant_types<Index + 1, TupleType, Func>(TupleType{}, func);
    }
}

using AllOperatorTypes = VariantTypes<OperatorConfigurationVariant>::type;

template<typename Func>
constexpr void apply_to_all_operators(Func func) {
    apply_to_variant_types(AllOperatorTypes{}, func);
}

struct OperatorHostConfiguration {
  bool enabled = true;

  std::vector<OperatorConfigurationVariant> operators;

  DERIVE_SERDE(OperatorHostConfiguration,
	       (&Self::enabled, "enabled")
	       (&Self::operators, "operators"))

  using MemberTypes = pc::reflect::type_list<bool, std::vector<OperatorConfigurationVariant>>;
  static const std::size_t MemberCount = 2;
};

} // namespace pc::operators
