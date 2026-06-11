#include "device_plugin.h"

#include <logger/logger.h>
#include <memory>
#include <pipeline/concurrent_operator_pipeline.h>
#include <plugins/backend/cpu/cpu_backend.h>
#include <plugins/operators/operator_plugin.h>
#include <plugins/operators/operator_variants.h>
#include <unordered_map>
#include <vector>
#include <workspace/workspace.h>

namespace pc::devices {

bool DevicePlugin::active() {
  return std::visit(
      [this](auto &device_config) {
        return effective_active(_workspace->config, device_config.id);
      },
      _config);
}

bool DevicePlugin::rendering() {
  return std::visit(
      [this](auto &device_config) {
        return effective_render(_workspace->config, device_config.id);
      },
      _config);
}

void DevicePlugin::on_config_field_changed(std::string_view path) {
  // For operator changes, forward the new config to all pipeline worker
  // instances
  if (path.find("operator") != std::string_view::npos) {
    std::visit(
        [this](const auto &device_config) {
          if constexpr (requires { device_config.operators; }) {
            for (size_t i = 0; i < device_config.operators.size(); i++) {
              const auto &operator_config = device_config.operators[i];
              for (const auto &worker_chain : _pipeline->worker_chains()) {
                worker_chain[i]->update_config(operator_config);
              }
            }
          }
        },
        _config);
  }
}

} // namespace pc::devices