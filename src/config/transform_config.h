#pragma once

#include <pointcaster/core_types.h>
#include <rfl/DefaultVal.hpp>

namespace pc {

struct TransformConfiguration {
  rfl::DefaultVal<float3> position = float3(0, 0, 0); // @minmax(-10, 10)
  rfl::DefaultVal<float3> rotation = float3(0, 0, 0); // @minmax(-360, 360)
  rfl::DefaultVal<float3> scale = float3(1, 1, 1);    // @minmax(0, 2.5)

  rfl::DefaultVal<float3> input_translation = float3(0, 0, 0); // @minmax(-10, 10)

  rfl::DefaultVal<float3> min_bound = float3(-10, -10, -10); // @minmax(-10, 10)
  rfl::DefaultVal<float3> max_bound = float3(10, 10, 10);    // @minmax(-10, 10)

  enum class BackendType { CPU, CUDA };
  rfl::DefaultVal<TransformConfiguration::BackendType> backend =
      BackendType::CPU;
};

} // namespace pc