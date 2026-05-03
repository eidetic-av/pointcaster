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

  void project_transform_frame_data(
      UShortDepthData input_depth_frame, RgbColorData input_rgb_frame,
      std::shared_ptr<PointCloud> output_cloud,
      const CameraIntrinsics &color_intrinsics,
      const TransformConfiguration &transform,
      const ColorTransformConfiguration &color_transform,
      std::span<std::byte> render_output = {}) override;
};

} // namespace pc::backend