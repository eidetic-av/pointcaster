#include "cuda_backend.h"
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

} // namespace pc::backend