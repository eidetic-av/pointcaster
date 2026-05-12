#include "fringe_removal_operator.h"
#include "fringe_removal_config.h"

#include <algorithm>
#include <camera/indexing_color_camera.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// #include <profiling/profiling_zone.h>

namespace pc::operators {

void FringeRemovalOperator::init() {
  pc::logger()->trace("Initialised FringeRemovalOperator");
}

void FringeRemovalOperator::process(
    const PointCloud &input, PointCloud &output,
    const OperatorConfigurationVariant &config_variant) {
  OperatorPlugin::process(input, output, config_variant);

  // TODO maybe profiling not working in dynamic plugin
  // profiling::ProfilingZone process_zone("FringeRemovalOperator::process");

  const auto &config = std::get<FringeRemovalConfiguration>(config_variant);

  // --- Test camera (typical depth-cam-like intrinsics) ---
  // TODO cam stuff should be in src/camera
  pc::camera::IndexingColorCamera cam;
  cam.fx = 554.0f;
  cam.fy = 554.0f;
  cam.width = 640 / 2;
  cam.height = 576 / 2;
  cam.cx = cam.width / 2.0f;
  cam.cy = cam.height / 2.0f;

  cam.extrinsic = extrinsic_from_camera_config(config.camera);

  // TODO check these unitx vs default
  // Identity + 2m translation along Z (camera 2m behind origin looking +Z)
  cam.extrinsic.values[11] = 2000.0f;

  _input_image = {.name = "Input", .frame_data = cam.project(input)};

  // TODO do pcl here
  // - show projection camera in qml canvas / image (X)
  // - add camera transform to the config (1/2)
  // - show in qml view3d
  // - do edge refinement filters
  // - apply to point cloud in 3d space
}

std::vector<pc::camera::CameraFrameRef> FringeRemovalOperator::camera_frames() {
  std::vector<pc::camera::CameraFrameRef> frames;
  frames.push_back(std::ref(_input_image));
  return frames;
}

} // namespace pc::operators

CORRADE_PLUGIN_REGISTER(FringeRemovalOperator,
                        pc::operators::FringeRemovalOperator,
                        "net.pointcaster.OperatorPlugin/1.0")
