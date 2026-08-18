#pragma once

#include <config/core_types_reflection.h>
#include <plugins/backend/backend_types.h>
#include <pointcaster/core.h>
#include <pointcaster/core_types.h>
#include <rfl/DefaultVal.hpp>

namespace pc {

struct TransformConfiguration {
  rfl::DefaultVal<float3> position = float3(0, 0, 0); // @minmax(-10, 10)
  rfl::DefaultVal<float3> rotation = float3(0, 0, 0); // @minmax(-360, 360)
  rfl::DefaultVal<float3> scale = float3(1, 1, 1);    // @minmax(0, 2.5)

  rfl::DefaultVal<bool> crop_to_bounds = false;
  rfl::DefaultVal<position_bounds> bounds = pc::default_config_bounds;

  rfl::DefaultVal<int> sample = 1; // @minmax(1, 64)

  rfl::DefaultVal<BackendType> backend = BackendType::CPU;
};

POINTCASTER_CORE_EXPORT float4x4 to_float4x4(const TransformConfiguration &t);

} // namespace pc