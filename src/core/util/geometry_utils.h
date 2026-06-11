#pragma once

#include <pointcaster/core.h>
#include <pointcaster/core_types.h>

namespace pc {

POINTCASTER_CORE_EXPORT float4x4 multiply(const float4x4 &a, const float4x4 &b);

POINTCASTER_CORE_EXPORT float4x4 translation_matrix(const float3 &t);
POINTCASTER_CORE_EXPORT float4x4 scale_matrix(const float3 &s);
POINTCASTER_CORE_EXPORT float4x4 rotation_matrix(const float3 &euler_degrees);

struct decomposed_transform {
  float3 position;
  quaternion rotation;
};

POINTCASTER_CORE_EXPORT
decomposed_transform decompose_transform(const float4x4 &matrix);

} // namespace pc