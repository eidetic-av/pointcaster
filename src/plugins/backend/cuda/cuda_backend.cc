#include "cuda_backend.h"
#include "cuda_kernels.h"
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <core/logger/logger.h>

namespace pc::backend {

CudaBackend::CudaBackend(Corrade::PluginManager::AbstractManager &manager,
                         Corrade::Containers::StringView plugin)
    : BackendPlugin(manager, plugin) {
  pc::logger()->trace("Initialised CUDA backend");
};

CudaBackend::~CudaBackend() {
  pc::logger()->trace("Destroyed CUDA backend");
};

void CudaBackend::transform_point_cloud(
    const TransformConfiguration &transform, PointCloud &output_cloud,
    const PointGeneratorFunction &generate_point) {
  cuda::transform_point_cloud(transform, output_cloud, generate_point);
};

} // namespace pc::backend

CORRADE_PLUGIN_REGISTER(CudaBackend, pc::backend::CudaBackend,
                        "net.pointcaster.BackendPlugin/1.0")