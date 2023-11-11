#pragma once
#include "../serialization.h"
#include "operator_types.h"

namespace pc::operators {

struct KNNFilterOperatorConfiguration {

  bool unfolded = false;
  bool enabled = true;

  DERIVE_SERDE(KNNFilterOperatorConfiguration,
	       (&Self::unfolded, "unfolded")
	       (&Self::enabled, "enabled"))

  using MemberTypes = pc::reflect::type_list<bool, bool>;
  static const std::size_t MemberCount = 2;

  static constexpr const char* Name = "KNN Filter";
};

struct KNNFilterOperator : public thrust::unary_function<point_t, point_t> {

  KNNFilterOperatorConfiguration _config;

  KNNFilterOperator(const KNNFilterOperatorConfiguration &config)
      : _config(config){};

  __device__ bool operator()(point_t point) const;

  static void draw_imgui_controls(KNNFilterOperatorConfiguration &config);
};

} // namespace pc::operators
