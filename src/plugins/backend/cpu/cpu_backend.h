#pragma once

#include "../backend_plugin.h"

namespace pc::backend {

class CpuBackend : public BackendPlugin {
public:
  explicit CpuBackend(Corrade::PluginManager::AbstractManager &manager,
                      Corrade::Containers::StringView plugin);

  ~CpuBackend();

  CpuBackend(const CpuBackend &) = delete;
  CpuBackend &operator=(const CpuBackend &) = delete;
  CpuBackend(CpuBackend &&) = delete;
  CpuBackend &operator=(CpuBackend &&) = delete;

  void project_transform_frame_data(
      UShortDepthData input_depth_frame, RgbColorData input_rgb_frame,
      PointCloud &output_cloud,
      const CameraIntrinsics &camera_intrinsics) override;
};

} // namespace pc::backend