#include "concurrent_operator_pipeline.h"

#include <logger/logger.h>

namespace pc::pipeline {

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