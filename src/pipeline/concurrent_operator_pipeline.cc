#include "concurrent_operator_pipeline.h"

#include <logger/logger.h>
#include <workspace/workspace.h>

namespace pc::pipeline {

std::vector<OperatorPipelineWorkerChain> build_worker_chains(
    std::span<const operators::OperatorConfigurationVariant> configs,
    size_t concurrency, operators::OperatorHost &owner, Workspace &workspace) {

  std::vector<OperatorPipelineWorkerChain> chains(concurrency);

  for (auto &chain : chains) {
    chain.reserve(configs.size());
    for (const auto &variant : configs) {
      const auto [operator_id, plugin_name] =
          operators::operator_info_from_variant(variant);

      auto op = workspace.operator_plugin_manager->instantiate(plugin_name);
      op->update_config(variant);
      op->init(&owner, *workspace.backend_plugin_manager);

      chain.push_back(std::unique_ptr<operators::OperatorPlugin>(op.release()));
    }
  }

  return chains;
}

void ConcurrentOperatorPipeline::start() {
  _latest.store(nullptr);
  for (size_t i = 0; i < _worker_chains.size(); i++) {
    _worker_threads.emplace_back(
        [this, i](auto stop_token) { worker_loop(i, stop_token); });
  }
}

void ConcurrentOperatorPipeline::stop() {
  _worker_threads.clear();
}

void ConcurrentOperatorPipeline::worker_loop(size_t worker_index,
                                             std::stop_token stop_token) {
  auto &chain = _worker_chains[worker_index];

  std::stop_callback on_stop(stop_token, [this] { _input_queue.abort(); });

  try {
    while (!stop_token.stop_requested()) {

      std::shared_ptr<PointCloud> input;

      _input_queue.pop(input);

      if (stop_token.stop_requested()) break;

      auto current = input;
      for (auto &op : chain) current = op->process(*current);

      _latest.store(current, std::memory_order_release);

      auto snapshot = std::make_shared<std::vector<camera::CameraFrame>>();
      for (auto &op : chain) {
        for (auto &ref : op->camera_frames()) {
          snapshot->push_back(ref.get());
        }
      }
      _latest_camera_frames.store(std::move(snapshot),
                                  std::memory_order_release);

      if (_on_complete) _on_complete(current);
    }

  } catch (const tbb::user_abort &) {
    // expected when aborting the tbb input queue
  }
}

} // namespace pc::pipeline