#pragma once
#include "../serialization.h"
#include "operator_types.h"

namespace pc::operators {

struct RotateOperatorConfiguration {

  bool unfolded = false;
  bool enabled = true;
  pc::types::Float3 euler_angles{0, 0, 0};

  DERIVE_SERDE(RotateOperatorConfiguration,
	       (&Self::unfolded, "unfolded")
	       (&Self::enabled, "enabled")
	       (&Self::euler_angles, "euler_angles"))

  using MemberTypes = pc::reflect::type_list<bool, bool, pc::types::Float3>;
  static const std::size_t MemberCount = 3;
};

struct RotateOperator : public thrust::unary_function<point_t, point_t> {

  RotateOperatorConfiguration _config;

  RotateOperator(const RotateOperatorConfiguration &config)
      : _config(config){};

  __device__ point_t operator()(point_t point) const;

  static void draw_imgui_controls(RotateOperatorConfiguration& config);
};

} // namespace pc::operators
