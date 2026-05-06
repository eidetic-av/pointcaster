#pragma once

#include <pointcaster/core_types.h>
#include <span>

namespace pc::registration {

// Compute rigid transform that maps source points onto target points.
// Minimum 3 pairs.
pc::float4x4 compute_rigid_transform(std::span<const pc::float3> source,
                                     std::span<const pc::float3> target);

} // namespace pc::registration