#pragma once
#include <Corrade/Containers/Pointer.h>
#include <memory>
#include <mutex>
#include <plugins/operators/operator_variants.h>
#include <pointcaster/point_cloud.h>
#include <span>
#include <string_view>
#include <vector>

namespace pc {
class Workspace;
}
namespace pc::pipeline {
class ConcurrentOperatorPipeline;
}

namespace pc::operators {

class OperatorPlugin;

class OperatorHost {
public:
  virtual ~OperatorHost();
  // Ask the host to re-run its operator pipeline on the current input.
  virtual void reprocess() = 0;

  std::vector<Corrade::Containers::Pointer<OperatorPlugin>> operators{};

  void feed_operator_pipeline(std::shared_ptr<PointCloud> cloud);

  void update_operator_in_pipeline(const OperatorConfigurationVariant &config,
                                   std::string_view changed_path);

protected:
  Workspace *_workspace = nullptr;

  // each host has a multi-threaded pipeline of operators.
  // _pipeline_mutex serialises pipeline lifetime (rebuild) against feeds and
  // in-flight config updates that come from other threads.
  std::mutex _pipeline_mutex;
  std::unique_ptr<pipeline::ConcurrentOperatorPipeline> _pipeline;

  // Reconcile `operators` against `configs` (host supplies them: devices visit
  // their variant, sessions pass their operator vector). Rebuilds the pipeline
  // if the operator set or the concurrency changed. Must be called on the same
  // thread that owns the host's configuration (the UI / apply path).
  void sync_operators(std::span<const OperatorConfigurationVariant> configs);
  void rebuild_pipeline(std::span<const OperatorConfigurationVariant> configs);

  // TODO
  // concurrency to something based on the amount of operators and acceptable
  // latency OR
  // it could be set directly by user
  virtual size_t pipeline_concurrency() const { return 4; }

  virtual void on_pipeline_output(std::shared_ptr<PointCloud>) {}
};

} // namespace pc::operators