#pragma once
#include "../serialization.h"
#include "operator_types.h"

namespace pc::operators {

struct RakeOperatorConfiguration {

  bool unfolded = false;
  bool enabled = true;

  pc::types::MinMax<short> depth_min_max {0, 0};
  pc::types::MinMax<short> height_min_max {0, 0};

  DERIVE_SERDE(RakeOperatorConfiguration,
	       (&Self::unfolded, "unfolded")
	       (&Self::enabled, "enabled")
	       (&Self::depth_min_max, "depth_min_max")
	       (&Self::height_min_max, "height_min_max"))

  using MemberTypes = pc::reflect::type_list<bool, bool, pc::types::MinMax<short>, pc::types::MinMax<short>>;
  static const std::size_t MemberCount = 4;
};

struct RakeOperator : public thrust::unary_function<point_t, point_t> {

  RakeOperatorConfiguration _config;

  RakeOperator(const RakeOperatorConfiguration &config)
      : _config(config){};

  __device__ point_t operator()(point_t point) const;

  static void draw_imgui_controls(RakeOperatorConfiguration& config);
};

} // namespace pc::operators
