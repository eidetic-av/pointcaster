#pragma once

#include "plugins/operators/operator_host.h"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <plugins/operators/operator_plugin.h>
#include <vector>

// when TBB is transitively included in Qt UI units, it really hates having
// Qt's "emit" defined
#pragma push_macro("emit")
#undef emit
#include <oneapi/tbb/concurrent_queue.h>
#pragma pop_macro("emit")

namespace pc::pipeline {

using OperatorPipelineWorkerChain =
    std::vector<std::unique_ptr<operators::OperatorPlugin>>;

std::vector<OperatorPipelineWorkerChain> build_worker_chains(
    std::span<const operators::OperatorConfigurationVariant> configs,
    size_t concurrency, operators::OperatorHost &owner, Workspace &workspace);

class ConcurrentOperatorPipeline {
public:
  // queue capacity defaults to the worker count...
  // Submitting a frame while the queue is full evicts the oldest queued frame
  ConcurrentOperatorPipeline(
      std::vector<OperatorPipelineWorkerChain> worker_chains,
      size_t max_queue_size = 0)
      : _worker_chains(std::move(worker_chains)) {
    const size_t cap = max_queue_size
                           ? max_queue_size
                           : std::max<size_t>(1, _worker_chains.size());
    _input_queue.set_capacity(cap);
  }
  ~ConcurrentOperatorPipeline() { stop(); }
  ConcurrentOperatorPipeline(const ConcurrentOperatorPipeline &) = delete;
  ConcurrentOperatorPipeline &
  operator=(const ConcurrentOperatorPipeline &) = delete;
  ConcurrentOperatorPipeline(ConcurrentOperatorPipeline &&) = delete;
  ConcurrentOperatorPipeline &operator=(ConcurrentOperatorPipeline &&) = delete;

  void start();
  void stop();

  // Assumes ownership of the submitted shared_ptr. Never blocks the producer.
  bool submit(std::shared_ptr<PointCloud> raw) {
    if (!raw) return false;
    PipelineFrame frame{_input_seq.fetch_add(1, std::memory_order_relaxed) + 1,
                        std::move(raw)};
    while (!_input_queue.try_push(frame)) {
      PipelineFrame dropped;
      _input_queue.try_pop(dropped); // evict oldest, then retry
    }
    return true;
  }

  std::shared_ptr<PointCloud> latest_cloud() const {
    return _latest.load(std::memory_order_acquire);
  }

  const std::vector<std::vector<std::unique_ptr<operators::OperatorPlugin>>> &
  worker_chains() {
    return _worker_chains;
  }

  void set_on_complete(std::function<void(std::shared_ptr<PointCloud>)> cb) {
    _on_complete = std::move(cb);
  }

  std::vector<camera::CameraFrame> latest_camera_frames() const {
    auto snap = _latest_camera_frames.load(std::memory_order_acquire);
    return snap ? *snap : std::vector<camera::CameraFrame>{};
  }

private:
  struct PipelineFrame {
    uint64_t seq = 0;
    std::shared_ptr<PointCloud> cloud;
  };

  std::vector<std::vector<std::unique_ptr<operators::OperatorPlugin>>>
      _worker_chains;

  tbb::concurrent_bounded_queue<PipelineFrame> _input_queue;

  std::atomic<std::shared_ptr<PointCloud>> _latest;
  std::atomic<std::shared_ptr<const std::vector<camera::CameraFrame>>>
      _latest_camera_frames;

  // monotonic ordering: seq assigned at submit, gate at publish
  std::atomic<uint64_t> _input_seq{0};
  std::atomic<uint64_t> _published_seq{0};

  std::function<void(std::shared_ptr<PointCloud>)> _on_complete;
  std::vector<std::jthread> _worker_threads;

  void worker_loop(size_t worker_index, std::stop_token stop_token);
};

} // namespace pc::pipeline