#pragma once
#include "../operator.h"

namespace pc::operators::cuda {

using pc::types::Float3;
using uid = unsigned long int;

struct TranslateOperatorConfiguration {
  uid id;
  bool unfolded = false;
  bool enabled = true;
  Float3 offset; // @minmax(-10, 10)
};

struct TranslateOperator : public thrust::unary_function<indexed_point_t, indexed_point_t> {

  TranslateOperatorConfiguration _config;

  TranslateOperator(const TranslateOperatorConfiguration &config)
      : _config(config){};

  __device__ indexed_point_t operator()(indexed_point_t point) const;
};

} // namespace pc::operators
