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

void DevicePlugin::on_config_field_changed(std::string_view path) {
  // For operator changes, forward the new config to all pipeline worker
  // instances
  if (path.find("operator") != std::string_view::npos) {
    std::visit(
        [this](const auto &device_config) {
          for (size_t i = 0; i < device_config.operators.size(); i++) {
            const auto &operator_config = device_config.operators[i];
            for (const auto &worker_chain : _pipeline->worker_chains()) {
              worker_chain[i]->update_config(operator_config);
            }
          }
        },
        _config);
  }
}

void DevicePlugin::sync_operators() {
  std::vector<operators::OperatorConfigurationVariant> operator_configs;
  std::visit(
      [&](const auto &device_config) {
        operator_configs.assign(device_config.operators.begin(),
                                device_config.operators.end());
      },
      _config);

  std::unordered_map<std::string, std::size_t> existing_index_by_id;
  existing_index_by_id.reserve(operators.size());
  for (std::size_t i = 0; i < operators.size(); ++i) {
    if (!operators[i]) continue;
    const auto [id, _] =
        operators::operator_info_from_variant(operators[i]->config_variant());
    if (id.empty()) continue;
    existing_index_by_id.try_emplace(id, i);
  }

  std::vector<Corrade::Containers::Pointer<operators::OperatorPlugin>>
      new_operators;
  new_operators.reserve(operator_configs.size());

  for (auto &operator_variant : operator_configs) {
    auto [operator_id, operator_plugin_name] =
        operators::operator_info_from_variant(operator_variant);
    if (operator_id.empty()) {
      pc::logger()->warn("Operator config missing id; skipping");
      continue;
    }

    Corrade::Containers::Pointer<operators::OperatorPlugin> op_plugin;
    bool new_instance = false;

    auto it = existing_index_by_id.find(operator_id);
    if (it != existing_index_by_id.end()) {
      const auto idx = it->second;
      existing_index_by_id.erase(it);

      auto &existing_ptr = operators[idx];
      const auto [_, existing_plugin_name] =
          operators::operator_info_from_variant(existing_ptr->config_variant());

      if (existing_plugin_name == operator_plugin_name) {
        existing_ptr->update_config(operator_variant);
        op_plugin = std::move(existing_ptr);
      } else {
        pc::logger()->info("Operator id='{}' type '{}' -> '{}'", operator_id,
                           existing_plugin_name, operator_plugin_name);
        existing_ptr = nullptr;
        op_plugin = _workspace->operator_plugin_manager->instantiate(
            std::string(operator_plugin_name));
        new_instance = true;
      }
    } else {
      op_plugin = _workspace->operator_plugin_manager->instantiate(
          std::string(operator_plugin_name));
      new_instance = true;
    }

    op_plugin->update_config(operator_variant);
    if (new_instance) {
      op_plugin->init(this, *_workspace->backend_plugin_manager);
    }

    new_operators.push_back(std::move(op_plugin));
  }

  bool changed = (new_operators.size() != operators.size());
  if (!changed) {
    for (size_t i = 0; i < new_operators.size(); ++i) {
      if (new_operators[i].get() != operators[i].get()) {
        changed = true;
        break;
      }
    }
  }

  // remaining entries in existing_index_by_id are deletions; just drop
  operators = std::move(new_operators);

  if (changed || !_pipeline) {
    rebuild_pipeline();
  }
}

void DevicePlugin::rebuild_pipeline() {
  if (_pipeline) {
    _pipeline->stop();
    _pipeline.reset();
  }

  // TODO
  constexpr size_t concurrency = 4;
  // size_t operator_count = 0;
  // std::visit([&](auto &config) { operator_count = config.operators.size(); },
  //            _config);
  // concurrency to something based on the amount of operators and acceptable
  // latency

  using OperatorPipelineWorkerChain =
      std::vector<std::unique_ptr<operators::OperatorPlugin>>;
  std::vector<OperatorPipelineWorkerChain> pipeline_worker_chains;

  pipeline_worker_chains.clear();
  pipeline_worker_chains.resize(concurrency);

  const auto operator_initialisation_visitor = [&](auto &device_config) {
    for (size_t thread_index = 0; thread_index < concurrency; thread_index++) {
      auto &worker_chain = pipeline_worker_chains[thread_index];

      // we create <concurrency> instances of each operator attached to this
      // device
      for (auto &variant : device_config.operators) {
        auto [operator_id, plugin_name] =
            operators::operator_info_from_variant(variant);

        auto op = _workspace->operator_plugin_manager->instantiate(plugin_name);
        op->update_config(variant);
        op->init(this, *_workspace->backend_plugin_manager);

        // TODO need to set backend using config
        op->set_current_backend(BackendType::CPU);

        worker_chain.push_back(
            std::unique_ptr<operators::OperatorPlugin>(op.release()));
      }
    }
  };

  std::visit(operator_initialisation_visitor, _config);

  // TODO queue length???
  _pipeline = std::make_unique<pipeline::ConcurrentOperatorPipeline>(
      std::move(pipeline_worker_chains), /*queue=*/8);

  _pipeline->set_on_complete([this](std::shared_ptr<PointCloud> output_cloud) {
    on_pipeline_output(output_cloud);
  });

  _pipeline->start();
}

void DevicePlugin::feed_operator_pipeline(std::shared_ptr<PointCloud> cloud) {
  if (_pipeline) {
    bool success = _pipeline->submit(cloud);
    if (!success) {
      pc::logger()->error(
          "Failed to submit device pointcloud into operator pipeline");
    }
  }
}

void DevicePlugin::update_operator_in_pipeline(
    const operators::OperatorConfigurationVariant &config,
    std::string_view changed_path) {
  if (!_pipeline) return;
  auto [target_id, _] = operators::operator_info_from_variant(config);
  for (auto &chain : _pipeline->worker_chains()) {
    for (auto &op : chain) {
      auto [id, __] =
          operators::operator_info_from_variant(op->config_variant());
      if (id == target_id) {
        op->update_config(config);
        op->on_config_field_changed(changed_path);
      }
    }
  }
}

} // namespace pc::devices