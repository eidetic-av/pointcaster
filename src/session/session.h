#pragma once

#include "session_config.h"

#include <logger/logger.h>
#include <plugins/operators/operator_host.h>
#include <thread>

namespace pc {

class Workspace;

namespace operators {
class OperatorPlugin;
}

class Session : public operators::OperatorHost {
public:
  const std::string id;

  explicit Session(Workspace &workspace, const SessionConfiguration &config);

  void reprocess() override { pc::logger()->error("Reprocess..."); };
  void update_config(const SessionConfiguration &config);

protected:
  size_t pipeline_concurrency() const override;
  void on_pipeline_output(std::shared_ptr<PointCloud> cloud) override;

private:
  SessionConfiguration _config;
  std::jthread _update_thread;
};

} // namespace pc