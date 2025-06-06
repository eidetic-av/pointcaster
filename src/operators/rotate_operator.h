#pragma once
#include "../serialization.h"
#include "operator.h"

namespace pc::operators {

using pc::types::Float3;
using uid = unsigned long int;

struct RotateOperatorConfiguration {
  uid id;
  bool unfolded = false;
  bool enabled = true;
  Float3 euler_angles; // @minmax(-360, 360)
};

struct RotateOperator : public thrust::unary_function<indexed_point_t, indexed_point_t> {

  RotateOperatorConfiguration _config;

  RotateOperator(const RotateOperatorConfiguration &config)
      : _config(config){};

  __device__ indexed_point_t operator()(indexed_point_t point) const;

  static void draw_imgui_controls(RotateOperatorConfiguration& config);
};

} // namespace pc::operators
