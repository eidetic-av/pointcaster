#pragma once

#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <variant>

namespace pc {
// all types representable inside configs...
// TODO we need float2, float3, float4, int2, int3, position, distance etc.
using ConfigValue =
    std::variant<float, int, std::string, bool, position_bounds, radius,
                 PointCloudPtr, VoxelisedCloudPtr, AabbListPtr>;

} // namespace pc