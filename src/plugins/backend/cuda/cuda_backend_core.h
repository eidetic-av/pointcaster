#pragma once

#include "../backend_types.h"

#include <config/transform_config.h>
#include <memory>
#include <pointcaster/point_cloud.h>

namespace pc::backend::cuda {

bool init_device_memory(void *owner, const size_t point_count);
void free_device_memory(void *owner);

void project_transform_frame_data(void *owner,
                                  UShortDepthData input_depth_frame,
                                  RgbColorData input_rgb_frame,
                                  std::shared_ptr<PointCloud> output_cloud,
                                  const CameraIntrinsics &color_intrinsics,
                                  const TransformConfiguration &transform,
                                  std::span<std::byte> render_output = {});

} // namespace pc::backend::cuda