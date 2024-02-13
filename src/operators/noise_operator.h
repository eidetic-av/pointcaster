#pragma once
#include "../identifiable.h"
#include "../serialization.h"
#include "operator.h"

namespace pc::operators {

struct NoiseOperatorConfiguration {
  std::string id;
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

  NoiseOperator(const NoiseOperatorConfiguration &config) : _config{config} {};

  __device__ indexed_point_t operator()(indexed_point_t point) const;
};

} // namespace pc::operators
