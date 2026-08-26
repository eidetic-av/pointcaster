#pragma once
#include "pipeline_frame.h"

#include "plugins/operators/operator_host.h"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <plugins/operators/operator_plugin.h>
#include <vector>

namespace pc::operators {

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
      size_t max_queue_size = 0);

  ~ConcurrentOperatorPipeline();
  ConcurrentOperatorPipeline(const ConcurrentOperatorPipeline &) = delete;
  ConcurrentOperatorPipeline &
  operator=(const ConcurrentOperatorPipeline &) = delete;
  ConcurrentOperatorPipeline(ConcurrentOperatorPipeline &&) = delete;
  ConcurrentOperatorPipeline &operator=(ConcurrentOperatorPipeline &&) = delete;

  void start();
  void stop();

  // non-blocking. submit ownership of the PointCloud
  bool submit(std::shared_ptr<PointCloud> raw);

  // null until the first frame has been published
  PipelineFramePtr latest_frame() const {
    return _latest.load(std::memory_order_acquire);
  }

  std::shared_ptr<PointCloud> latest_cloud() const {
    auto frame = _latest.load(std::memory_order_acquire);
    return frame ? frame->cloud : nullptr;
  }

  const std::vector<std::vector<std::unique_ptr<operators::OperatorPlugin>>> &
  worker_chains() {
    return _worker_chains;
  }

  void set_on_complete(std::function<void(PipelineFramePtr)> cb) {
    _on_complete = std::move(cb);
  }

  std::vector<camera::CameraFrame> latest_camera_frames() const {
    auto snap = _latest_camera_frames.load(std::memory_order_acquire);
    return snap ? *snap : std::vector<camera::CameraFrame>{};
  }

private:
  std::vector<std::vector<std::unique_ptr<operators::OperatorPlugin>>>
      _worker_chains;

  struct InputQueue;
  std::unique_ptr<InputQueue> _input_queue;

  std::atomic<PipelineFramePtr> _latest;

  std::atomic<std::shared_ptr<const std::vector<camera::CameraFrame>>>
      _latest_camera_frames;

  // monotonic ordering: seq assigned at submit, gate at publish
  std::atomic<uint64_t> _input_seq{0};
  std::atomic<uint64_t> _published_seq{0};

  std::function<void(PipelineFramePtr)> _on_complete;
  std::vector<std::jthread> _worker_threads;

  void worker_loop(size_t worker_index, std::stop_token stop_token);
};

} // namespace pc::operators