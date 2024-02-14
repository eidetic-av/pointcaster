#pragma once
#include "../serialization.h"
#include "../structs.h"
#include "operator.h"

namespace pc::operators {

using pc::types::Int3;

struct AABB {
  bool unfolded = true;
  Int3 min{-10000, -10000, -10000}; // @minmax(-10000, 10000)
  Int3 max{10000, 10000, 10000}; // @minmax(-10000, 10000)
};

using uid = unsigned long int;

struct RangeFilterFillConfiguration {
  int input_count;
  int max_fill = 5000;
  float fill_value;
  bool publish = false;
};

struct RangeFilterOperatorConfiguration {
  uid id;
  bool enabled = true;
  bool bypass = false;
  AABB aabb;
  RangeFilterFillConfiguration fill;
};

struct RangeFilterOperator : Operator {

  RangeFilterOperatorConfiguration _config;

  RangeFilterOperator(const RangeFilterOperatorConfiguration &config)
      : _config(config){};

  __device__ bool operator()(indexed_point_t point) const;

  static void draw_imgui_controls(RangeFilterOperatorConfiguration &config);
};

} // namespace pc::operators
