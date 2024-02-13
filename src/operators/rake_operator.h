#pragma once
#include "../serialization.h"
#include "operator.h"

namespace pc::operators {

struct RakeOperatorConfiguration {
  std::string id;
  bool unfolded = false;
  bool enabled = true;
  pc::types::MinMax<short> depth_min_max {0, 0};
  pc::types::MinMax<short> height_min_max {0, 0};
};

struct RakeOperator : public thrust::unary_function<indexed_point_t, indexed_point_t> {

  RakeOperatorConfiguration _config;

  RakeOperator(const RakeOperatorConfiguration &config)
      : _config(config){};

  __device__ indexed_point_t operator()(indexed_point_t point) const;

  static void draw_imgui_controls(RakeOperatorConfiguration& config);
};

} // namespace pc::operators
