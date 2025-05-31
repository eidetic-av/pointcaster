#pragma once
#include "../operator.h"

namespace pc::operators::cuda {

using pc::types::Float3;
using uid = unsigned long int;

struct UniformGainOperatorConfiguration {
  uid id;
  bool unfolded = false;
  bool enabled = true;
  float gain; // @minmax(-1, 1)
};

struct UniformGainOperator : public thrust::unary_function<indexed_point_t, indexed_point_t> {

  UniformGainOperatorConfiguration _config;

  explicit UniformGainOperator(const UniformGainOperatorConfiguration &config)
      : _config(config) { };

  __device__ indexed_point_t operator()(indexed_point_t point) const;
};

} // namespace pc::operators
