#pragma once

#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <variant>

namespace pc {
// all types representable inside configs...
using ConfigValue =
    std::variant<bool, int, float, double, std::string, position_bounds, radius,
                 PointCloudPtr, VoxelisedCloudPtr, AabbListPtr>;

} // namespace pc