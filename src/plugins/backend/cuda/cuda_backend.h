#pragma once

#include "../backend_plugin.h"

namespace pc::backend {

class CudaBackend : public BackendPlugin {
public:
  explicit CudaBackend(Corrade::PluginManager::AbstractManager &manager,
                       Corrade::Containers::StringView plugin);

  ~CudaBackend();

  CudaBackend(const CudaBackend &) = delete;
  CudaBackend &operator=(const CudaBackend &) = delete;
  CudaBackend(CudaBackend &&) = delete;
  CudaBackend &operator=(CudaBackend &&) = delete;

  void
  transform_point_cloud(const TransformConfiguration &transform,
                        PointCloud &output_cloud,
                        const PointGeneratorFunction &generate_point) override;
};

} // namespace pc::backend