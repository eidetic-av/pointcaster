#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
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

inline int16_t to_position_axis(float value) {
  constexpr auto lowest =
      static_cast<float>(std::numeric_limits<int16_t>::lowest());
  constexpr auto highest =
      static_cast<float>(std::numeric_limits<int16_t>::max());
  return static_cast<int16_t>(std::clamp(value, lowest, highest));
}

inline position to_position(float x, float y, float z) {
  return {to_position_axis(x), to_position_axis(y), to_position_axis(z)};
}

} // namespace pc