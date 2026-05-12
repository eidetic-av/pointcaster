#include "indexing_color_camera.h"
#include "camera_frame.h"

#include <logger/logger.h>
#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>

namespace pc::camera {

// TODO
// maybe this camera stuff can be a static plugin, and then operators that need
// it can link to that

CameraFrameData IndexingColorCamera::project(const PointCloud &cloud) const {
  CameraFrameData result;
  result.width = width;
  result.height = height;

  auto &color_buffer = result.color_buffers["color"];
  auto &depth_buffer = result.float_buffers["depth"];
  auto &index_buffer = result.int_buffers["index"];

  // TODO clear color
  color_buffer.assign(width * height, color{30, 30, 30, 255});
  depth_buffer.assign(width * height, std::numeric_limits<float>::infinity());
  index_buffer.assign(width * height, -1);

  const auto &m = extrinsic.values;

  // TODO i think this should be an async gpu task not a serial for loop on
  // the CPU

  for (int i = 0; i < static_cast<int>(cloud.size()); ++i) {
    const auto &p = cloud.positions[i];

    auto wx = static_cast<float>(p.x);
    auto wy = static_cast<float>(p.y);
    auto wz = static_cast<float>(p.z);

    // transform to camera space (row-major multiply)
    float cam_x = m[0] * wx + m[1] * wy + m[2] * wz + m[3];
    float cam_y = m[4] * wx + m[5] * wy + m[6] * wz + m[7];
    float cam_z = m[8] * wx + m[9] * wy + m[10] * wz + m[11];

    if (cam_z <= 0.0f) continue;

    float u = fx * (cam_x / cam_z) + cx;
    float v = fy * (cam_y / cam_z) + cy;

    int ui = static_cast<int>(std::round(u));
    int vi = static_cast<int>(std::round(v));

    if (ui < 0 || ui >= width || vi < 0 || vi >= height) continue;

    int idx = vi * width + ui;
    if (cam_z < depth_buffer[idx]) {
      depth_buffer[idx] = cam_z;
      index_buffer[idx] = i;
      auto c = cloud.colors[i];
      c.a = 255;
      color_buffer[idx] = c;
    }
  }

  int hit_count = 0;
  for (auto idx : index_buffer)
    if (idx >= 0) hit_count++;
  pc::logger()->debug("IndexingColorCam: {}/{} pixels hit", hit_count,
                      result.width * result.height);

  return result;
}

} // namespace pc::camera