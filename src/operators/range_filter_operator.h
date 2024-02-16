#pragma once
#include "../serialization.h"
#include "../structs.h"
#include "operator.h"

namespace pc::operators {

using pc::types::Float3;

struct AABB {
  bool unfolded = true;
  Float3 min{-10000, -10000, -10000}; // @minmax(-10000, 10000)
  Float3 max{10000, 10000, 10000}; // @minmax(-10000, 10000)
};

using uid = unsigned long int;

struct RangeFilterOperatorFillConfiguration {
  int fill_count;
  int max_fill = 5000;
  float fill_value;
  float proportion;
  bool publish = false;
};

struct RangeFilterOperatorConfiguration {
  uid id;
  bool enabled = true;
  bool bypass = false;
  Float3 position {0, 0, 0}; // @minmax(-10, 10)
  Float3 size {1, 1, 1}; // @minmax(0.01f, 10)
  RangeFilterOperatorFillConfiguration fill;
};

struct RangeFilterOperator : Operator {

  RangeFilterOperatorConfiguration _config;

  RangeFilterOperator(const RangeFilterOperatorConfiguration &config)
      : _config(config){};

  __device__ bool operator()(indexed_point_t point) const;

};

} // namespace pc::operators
