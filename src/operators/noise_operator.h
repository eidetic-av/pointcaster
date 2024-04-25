#pragma once
#include "../identifiable.h"
#include "operator.h"

namespace pc::operators {

using uid = unsigned long int;

struct NoiseOperatorConfiguration {
  uid id;
  bool enabled = true;
  float scale = 0.001f;
  float magnitude = 500.0f;
  int seed = 99;
  int repeat = 32;
  float lacunarity = 2.0f;
  float decay = 0.5f;
};

struct NoiseOperator : public Operator {

  NoiseOperatorConfiguration _config;

  NoiseOperator(const NoiseOperatorConfiguration &config);
  
  __device__ indexed_point_t operator()(indexed_point_t point) const;
};

} // namespace pc::operators
