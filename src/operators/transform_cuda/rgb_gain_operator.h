#pragma once
#include "../operator.h"

namespace pc::operators::cuda {

using pc::types::Float3;
using uid = unsigned long int;

struct RGBGainOperatorConfiguration {
  uid id;
  bool unfolded = false;
  bool enabled = true;
  Float3 gain; // @minmax(-1, 1)
  float multiplier = 1; // @minmax(-2, 2)
};

struct RGBGainOperator : public thrust::unary_function<indexed_point_t, indexed_point_t> {

  RGBGainOperatorConfiguration _config;

  RGBGainOperator(const RGBGainOperatorConfiguration &config)
      : _config(config) { };

  __device__ indexed_point_t operator()(indexed_point_t point) const;
};

} // namespace pc::operators
