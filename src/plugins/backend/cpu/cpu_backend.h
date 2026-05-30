#pragma once

#include "../backend_plugin.h"
#include <BS_thread_pool.hpp>

namespace pc::backend {

class CpuBackend : public BackendPlugin {
public:
  inline static BS::thread_pool thread_pool{};

  explicit CpuBackend(Corrade::PluginManager::AbstractManager &manager,
                      Corrade::Containers::StringView plugin);

  ~CpuBackend();

  CpuBackend(const CpuBackend &) = delete;
  CpuBackend &operator=(const CpuBackend &) = delete;
  CpuBackend(CpuBackend &&) = delete;
  CpuBackend &operator=(CpuBackend &&) = delete;

  void transform_point_cloud(
      const PointCloud &input_cloud, std::shared_ptr<PointCloud> output_cloud,
      const TransformConfiguration &transform,
      const ColorTransformConfiguration &color_transform) const override;

  void transform_point_cloud(
      std::span<const position> input_positions,
      std::span<const color> input_colors,
      std::shared_ptr<PointCloud> output_cloud,
      const TransformConfiguration &transform,
      const ColorTransformConfiguration &color_transform) const override;

  void project_transform_frame_data(
      std::span<const uint16_t> input_depth_frame,
      std::span<const color_rgb> input_rgb_frame,
      std::shared_ptr<PointCloud> output_cloud,
      const CameraIntrinsics &color_intrinsics,
      const TransformConfiguration &transform,
      const ColorTransformConfiguration &color_transform) const override;

  void pack_render_buffer(const PointCloud &cloud,
                          std::span<std::byte> output) const override;

  void project_frame(const PointCloud &cloud, camera::CameraFrameData &output,
                     camera::FrameProjectionArgs projection) const override;
};

} // namespace pc::backend