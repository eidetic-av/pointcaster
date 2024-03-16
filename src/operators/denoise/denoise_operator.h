#pragma once
#include "../../serialization.h"
#include "../../structs.h"
#include "../operator.h"
#include <Magnum/Magnum.h>
#include <Magnum/SceneGraph/SceneGraph.h>

namespace pc::operators {

using pc::types::Float2;
using pc::types::Float3;

using uid = unsigned long int;

struct DenoiseOperatorConfiguration {
  uid id;
  bool enabled = true;
  bool bypass = false;
  bool draw = true;
};

struct DenoiseOperator : Operator {

  DenoiseOperatorConfiguration _config;

  DenoiseOperator(const DenoiseOperatorConfiguration &config)
      : _config(config){};

  __device__ bool operator()(indexed_point_t point) const;
};

} // namespace pc::operators
