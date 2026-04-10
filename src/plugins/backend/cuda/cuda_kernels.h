#pragma once
#include <config/transform_config.h>
#include <pointcaster/point_cloud.h>

namespace pc::backend::cuda {

using PointType = std::tuple<position, color>;
using PointGeneratorFunction = std::function<PointType(const int)>;

void transform_point_cloud(const TransformConfiguration &transform,
                           PointCloud &output_cloud,
                           const PointGeneratorFunction &generate_point);

}