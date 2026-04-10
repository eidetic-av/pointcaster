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

  void
  transform_point_cloud(const TransformConfiguration &transform,
                        PointCloud &output_cloud,
                        const PointGeneratorFunction &generate_point) override;
};

} // namespace pc::backend