#pragma once

#include "../backend_types.h"

#include <config/color_transform_config.h>
#include <config/transform_config.h>
#include <memory>
#include <pointcaster/point_cloud.h>

namespace pc::backend::cuda {

bool init_device_memory(const void *owner, const size_t point_count);
void free_device_memory(const void *owner);

void project_transform_frame_data(
    const void *owner, std::span<const uint16_t> input_depth_frame,
    std::span<const color_rgb> input_rgb_frame,
    std::shared_ptr<PointCloud> output_cloud,
    const CameraIntrinsics &color_intrinsics,
    const TransformConfiguration &transform,
    const ColorTransformConfiguration &color_transform,
    std::span<std::byte> render_output = {});

} // namespace pc::backend::cuda