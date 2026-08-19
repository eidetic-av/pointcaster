#pragma once

#include "backend_plugin.h"
#include "backend_types.h"

#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/StringStlView.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/Manager.h>
#include <cstddef>
#include <logger/logger.h>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace pc::backend {

constexpr std::string_view backend_name(BackendType backend_type) {
  return backend_type == BackendType::CUDA ? "CUDA" : "CPU";
}

// the plugin name each backend type is published under
constexpr std::optional<BackendType>
backend_type_from_plugin_name(std::string_view plugin_name) {
  if (plugin_name == "CpuBackend") return BackendType::CPU;
  if (plugin_name == "CudaBackend") return BackendType::CUDA;
  return std::nullopt;
}

// holds one instance of every loaded backend plugin
class BackendSet {
public:
  // instantiates every backend plugin the manager has loaded
  void
  instantiate(Corrade::PluginManager::Manager<BackendPlugin> &plugin_manager,
              std::string_view owner_name) {
    using Corrade::PluginManager::LoadState;

    for (const auto &plugin_name : plugin_manager.pluginList()) {
      if (plugin_manager.loadState(plugin_name) & LoadState::NotLoaded) {
        continue;
      }
      const auto backend_type =
          backend_type_from_plugin_name(std::string_view{plugin_name});
      if (!backend_type) {
        pc::logger()->warn("Unrecognised backend plugin '{}'",
                           std::string_view{plugin_name});
        continue;
      }
      if (_backends.contains(*backend_type)) continue;

      auto backend = plugin_manager.instantiate(plugin_name);
      if (!backend) {
        pc::logger()->error("{} failed to instantiate {} backend", owner_name,
                            backend_name(*backend_type));
        continue;
      }
      backend->init();
      _backends.emplace(*backend_type, std::move(backend));

      pc::logger()->trace("{} created {} backend", owner_name,
                          backend_name(*backend_type));
    }
  }

  // corrade's Pointer propagates const to its callsite...
  // so we need to reimplement a get() that gives a mutable ptr
  BackendPlugin *get(BackendType backend_type) const {
    auto it = _backends.find(backend_type);
    if (it == _backends.end()) return nullptr;
    return const_cast<BackendPlugin *>(it->second.get());
  }

  BackendPlugin *cpu() const { return get(BackendType::CPU); }

  bool contains(BackendType backend_type) const {
    return _backends.contains(backend_type);
  }

  bool empty() const { return _backends.empty(); }

  std::optional<BackendType> resolve(BackendType requested_backend) const {
    if (_backends.contains(requested_backend)) return requested_backend;
    if (_backends.contains(BackendType::CPU)) return BackendType::CPU;
    if (_backends.empty()) return std::nullopt;
    return _backends.begin()->first;
  }

private:
  std::unordered_map<BackendType, Corrade::Containers::Pointer<BackendPlugin>>
      _backends;
};

} // namespace pc::backend
