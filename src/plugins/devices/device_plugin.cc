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

void DevicePlugin::update_config(const DeviceConfigurationVariant &config) {
  _config = config;
  std::visit(
      [this](auto &device_config) {
        if constexpr (requires { device_config.operators; }) {
          sync_operators(device_config.operators);
        }
      },
      _config);

  if (!_workspace) return;

  // we updated the plugin's config copy, now update the actual config thats
  // part of the workspace
  const auto device_id = device_id_from_variant(_config);
  std::string node_prefix;
  {
    std::scoped_lock lock(_workspace->config_access);
    for (auto &device_config_variant : _workspace->config.devices) {
      if (device_id_from_variant(device_config_variant) != device_id) continue;
      device_config_variant = _config;
      node_prefix = "device/" +
                    device_address(_workspace->config, std::string(device_id));
      break;
    }
  }
  if (node_prefix.empty()) return;

  _workspace->config_registry.notify(node_prefix + "/");
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