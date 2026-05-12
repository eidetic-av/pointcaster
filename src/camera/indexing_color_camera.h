#pragma once

#include "camera_frame.h"

#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <pointcaster_api.h>

namespace pc::camera {

class POINTCASTER_API IndexingColorCamera {
public:
  float fx, fy, cx, cy;
  int width, height;
  float4x4 extrinsic;

  CameraFrameData project(const PointCloud &cloud) const;
};
} // namespace pc::camera