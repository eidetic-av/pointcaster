#pragma once

#include <pointcaster/core.h>
#include <pointcaster/core_types.h>


namespace pc {

struct decomposed_transform {
  pc::float3 position;
  pc::quaternion rotation;
};

POINTCASTER_CORE_EXPORT
decomposed_transform decompose_transform(const pc::float4x4 &matrix);

} // namespace pc