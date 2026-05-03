#pragma once

#include "../backend_plugin.h"

namespace pc::backend {

class CudaBackend : public BackendPlugin {
public:
  explicit CudaBackend(Corrade::PluginManager::AbstractManager &manager,
                       Corrade::Containers::StringView plugin)
      : BackendPlugin(manager, plugin) {};

  ~CudaBackend();

  CudaBackend(const CudaBackend &) = delete;
  CudaBackend &operator=(const CudaBackend &) = delete;
  CudaBackend(CudaBackend &&) = delete;
  CudaBackend &operator=(CudaBackend &&) = delete;

  void init(const size_t point_count) override;

  void project_transform_frame_data(
      UShortDepthData input_depth_frame, RgbColorData input_rgb_frame,
      std::shared_ptr<PointCloud> output_cloud,
      const CameraIntrinsics &color_intrinsics,
      const TransformConfiguration &transform,
      const ColorTransformConfiguration &color_transform,
      std::span<std::byte> render_output = {}) override;
};

} // namespace pc::backend