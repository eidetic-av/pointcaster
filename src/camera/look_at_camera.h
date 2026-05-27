#pragma once

#include "camera_frame.h"
#include "look_at_camera_config.h"

#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <pointcaster_api.h>

namespace pc::backend {
class BackendPlugin;
}

namespace pc::camera {

class POINTCASTER_API LookAtCamera {
public:
  void update_config(const LookAtCameraConfiguration &camera_config);

  void project(const PointCloud &cloud, backend::BackendPlugin *backend);

  std::shared_ptr<CameraFrameData> result() { return _result; }

  FrameProjectionArgs projection_args() { return _projection_args; }

private:
  FrameProjectionArgs _projection_args;
  int _flood_fill_passes;
  std::shared_ptr<CameraFrameData> _result;
};
} // namespace pc::camera