#include "concurrent_operator_pipeline.h"
#include "plugins/operators/operator_variants.h"
#include <logger/logger.h>
#include <memory>
#include <profiling/profiling_zone.h>
#include <workspace/workspace.h>

namespace pc::operators {

using namespace pc::profiling;

std::vector<OperatorPipelineWorkerChain> build_worker_chains(
    std::span<const operators::OperatorConfigurationVariant> configs,
    size_t concurrency, operators::OperatorHost &owner, Workspace &workspace) {
  std::vector<OperatorPipelineWorkerChain> chains(concurrency);
  for (auto &chain : chains) {
    chain.reserve(configs.size());
    for (const auto &variant : configs) {
      const auto [operator_id, plugin_name] =
          operators::operator_info_from_variant(variant);
      if (!(workspace.operator_plugin_manager->loadState(plugin_name) &
            Corrade::PluginManager::LoadState::Loaded)) {
        pc::logger()->error("Operator plugin '{}' is not loaded; "
                            "skipping operator id='{}' in pipeline",
                            plugin_name, operator_id);
        continue;
      }
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
  _published_seq.store(0, std::memory_order_release);
  for (size_t i = 0; i < _worker_chains.size(); i++) {
    _worker_threads.emplace_back(
        [this, i](auto stop_token) { worker_loop(i, stop_token); });
  }
}

void ConcurrentOperatorPipeline::stop() {
  _worker_threads.clear();
}

bool ConcurrentOperatorPipeline::submit(std::shared_ptr<PointCloud> raw) {
  if (!raw) return false;
  PipelineFramePtr frame = std::make_shared<const PipelineFrame>(PipelineFrame{
      .seq = _input_seq.fetch_add(1, std::memory_order_relaxed) + 1,
      .cloud = std::move(raw),
      .additional_streams = {}});
  while (!_input_queue.try_push(frame)) {
    PipelineFramePtr dropped;
    _input_queue.try_pop(dropped); // evict oldest, then retry
  }
  return true;
}

void ConcurrentOperatorPipeline::worker_loop(size_t worker_index,
                                             std::stop_token stop_token) {
  auto &chain = _worker_chains[worker_index];
  std::stop_callback on_stop(stop_token, [this] { _input_queue.abort(); });
  try {
    while (!stop_token.stop_requested()) {
      PipelineFramePtr current;
      _input_queue.pop(current);
      if (stop_token.stop_requested()) break;
      if (!current) continue;

      const uint64_t seq = current->seq;

      // skip work that's already been superseded
      if (seq <= _published_seq.load(std::memory_order_acquire)) continue;

      {
        ProfilingZone zone("Operators::process");
        // the actual operator processing occurs here
        for (auto &op : chain) {
          if (!check_active(op->config_variant())) continue;
          current = op->process(current);
        }
      }
      if (!current) continue;

      // Monotonic output gate: only publish if this frame is strictly newer
      // than whatever has already been shown
      uint64_t prev = _published_seq.load(std::memory_order_acquire);
      bool is_newest = seq > prev;
      while (is_newest && !_published_seq.compare_exchange_weak(
                              prev, seq, std::memory_order_acq_rel,
                              std::memory_order_acquire)) {
        is_newest = seq > prev;
      }
      if (!is_newest) {
        // a newer frame already won; drop this result
        continue;
      }

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
  } catch (const std::exception &e) {
    pc::logger()->error("An operator threw the exception: {}", e.what());
    pc::logger()->error("Pipeline has been stopped");
  } catch (...) {
    pc::logger()->error("An operator threw an unknown exception...");
    pc::logger()->error("Pipeline has been stopped");
  }
}

} // namespace pc::operators