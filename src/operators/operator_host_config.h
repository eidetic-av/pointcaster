#pragma once

#include "../serialization.h"
#include "../structs.h"
#include "denoise/denoise_operator.gen.h"
#include "noise_operator.gen.h"
#include "operator.h"
#include "pcl/cluster_extraction_operator.gen.h"
#include "rake_operator.gen.h"
#include "range_filter_operator.gen.h"
#include "rotate_operator.gen.h"
#include "sample_filter_operator.gen.h"
#include "transform_cuda/translate_operator.gen.h"
#include <functional>
#include <variant>


namespace pc::operators {

using OperatorConfigurationVariant =
    std::variant<NoiseOperatorConfiguration, SampleFilterOperatorConfiguration,
                 RangeFilterOperatorConfiguration, RotateOperatorConfiguration,
                 cuda::TranslateOperatorConfiguration,
                 RakeOperatorConfiguration,
                 pcl_cpu::ClusterExtractionConfiguration>;

// Extract types from variant into a tuple
template <typename Variant> struct VariantTypes;

template <typename... Args> struct VariantTypes<std::variant<Args...>> {
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

template <typename Func> constexpr void apply_to_all_operators(Func func) {
  apply_to_variant_types(AllOperatorTypes{}, func);
}

struct OperatorHostConfiguration {
  bool enabled = true;
  std::vector<OperatorConfigurationVariant> operators;
};

using OperatorList = std::vector<OperatorHostConfiguration>;

extern operator_in_out_t apply(operator_in_out_t begin, operator_in_out_t end,
                               OperatorList &operator_list);

extern pc::types::PointCloud apply(const pc::types::PointCloud &point_cloud,
                                   OperatorList &operator_list);

extern pc::types::PointCloud
apply(const pc::types::PointCloud &point_cloud,
      std::vector<OperatorConfigurationVariant> &operator_variant_list);

} // namespace pc::operators
