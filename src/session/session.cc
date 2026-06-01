#include "session.h"
#include "session/session_config.h"
#include <memory>
#include <pipeline/concurrent_operator_pipeline.h>
#include <stop_token>
#include <workspace/workspace.h>

namespace pc {

namespace {
using namespace std::chrono;
using namespace std::chrono_literals;
constexpr auto period_for(int hz) {
  return duration_cast<steady_clock::duration>(
      duration<double>(1.0 / std::max(hz, 1)));
}

void update(std::stop_token stop_token, Workspace &workspace,
            std::string session_id) {

  auto next_tick = steady_clock::now();
  int update_hz = 240;

  while (!stop_token.stop_requested()) {
    {
      std::lock_guard lock(workspace.config_access);
      auto &sessions = workspace.config.sessions;
      auto it =
          std::ranges::find(sessions, session_id, &SessionConfiguration::id);
      // exit the update loop if the config for this session no longer exists in
      // the workspace
      if (it == sessions.end()) break;
      update_hz = it->operator_pipeline.value().update_hz.value();
    }
    const auto this_loop_period = period_for(update_hz);
    // do streaming of network based on config
    // TODO streaming per device
    // streaming of entire session
    next_tick += this_loop_period;
    next_tick = std::max(next_tick, steady_clock::now());
    std::this_thread::sleep_until(next_tick);
  }
}
} // namespace

Session::Session(Workspace &workspace, const SessionConfiguration &config)
    : id(config.id) {
  _workspace = &workspace;
  // build the initial operator set + pipeline on the construction thread
  update_config(config);
  _update_thread = std::jthread(update, std::ref(workspace), id);
}

void Session::update_config(const SessionConfiguration &config) {
  _config = config;
  sync_operators(_config.operators);
}

size_t Session::pipeline_concurrency() const {
  return size_t(_config.operator_pipeline.value().concurrency.value());
}

void Session::on_pipeline_output(std::shared_ptr<PointCloud> cloud) {
  // do streaming of network based on config
  // TODO streaming per device
  // streaming of entire session
}

} // namespace pc