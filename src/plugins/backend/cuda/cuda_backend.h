#pragma once

#include "../backend_plugin.h"

namespace pc::backend {

class CudaBackend : public BackendPlugin {

  explicit CudaBackend(Corrade::PluginManager::AbstractManager &manager,
                       Corrade::Containers::StringView plugin);

  ~CudaBackend();

  CudaBackend(const CudaBackend &) = delete;
  CudaBackend &operator=(const CudaBackend &) = delete;
  CudaBackend(CudaBackend &&) = delete;
  CudaBackend &operator=(CudaBackend &&) = delete;
};

} // namespace pc::backend