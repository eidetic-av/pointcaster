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
      std::span<const uint16_t> input_depth_frame,
      std::span<const color_rgb> input_rgb_frame,
      std::shared_ptr<PointCloud> output_cloud,
      const CameraIntrinsics &color_intrinsics,
      const TransformConfiguration &transform,
      const ColorTransformConfiguration &color_transform,
      std::span<std::byte> render_output = {}) const override;

  void transform_point_cloud(const PointCloud &input_cloud,
                             std::shared_ptr<PointCloud> output_cloud,
                             const TransformConfiguration &transform,
                             const ColorTransformConfiguration &color_transform,
                             std::span<std::byte> render_output) const override;

  void transform_point_cloud(std::span<const position> input_positions,
                             std::span<const color> input_colors,
                             std::shared_ptr<PointCloud> output_cloud,
                             const TransformConfiguration &transform,
                             const ColorTransformConfiguration &color_transform,
                             std::span<std::byte> render_output) const override;
};

} // namespace pc::backend