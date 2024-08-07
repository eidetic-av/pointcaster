#pragma once
#include "../serialization.h"
#include "operator.h"

namespace pc::operators {

using uid = unsigned long int;

struct SampleFilterOperatorConfiguration {
  uid id;
  bool unfolded = false;
  bool enabled = true;
  int sample_count = 1; // @minmax(1, 1000)
};

struct SampleFilterOperator : public thrust::unary_function<indexed_point_t, indexed_point_t> {

  SampleFilterOperatorConfiguration _config;

  SampleFilterOperator(const SampleFilterOperatorConfiguration &config)
      : _config(config){};

  __device__ bool operator()(indexed_point_t point) const;

  static void draw_imgui_controls(SampleFilterOperatorConfiguration &config);
};

} // namespace pc::operators
