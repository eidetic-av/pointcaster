#include "point_streamer.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <tbb/concurrent_vector.h>
#include <thread>
#include <workspace/workspace.h>
#include <workspace/workspace_config.h>

namespace pc::networking {

namespace {

using namespace std::chrono;
using namespace std::chrono_literals;

constexpr auto period_for(int hz) {
  return duration_cast<steady_clock::duration>(
      duration<double>(1.0 / std::max(hz, 1)));
}

void streaming_thread_loop(std::stop_token stop_token, Workspace &workspace) {

  // thread copy of our workspace config, snapshotted at the top of the loop
  pc::WorkspaceConfiguration workspace_config;

  auto next_tick = steady_clock::now();

  while (!stop_token.stop_requested()) {
    {
      std::lock_guard lock(workspace.config_access);
      workspace_config = workspace.config;
    }
    const auto &streamer_config = workspace_config.point_streamer.value();
    const auto this_loop_period =
        period_for(streamer_config.publish_hz.value());

    // do streaming of network based on config

    // TODO streaming per device

    // TODO streaming of entire session
    // SO THIS IS WHERE I WOULD THEN HAVE ACCESS TO MY SESSION POINT CLOUD ALSO,
    // SO I CAN USE IT TO STREAM USING ZMQ

    next_tick += this_loop_period;
    next_tick = std::max(next_tick, steady_clock::now());
    std::this_thread::sleep_until(next_tick);
  }
}
} // namespace

PointStreamer::PointStreamer(Workspace &workspace)
    : _worker(streaming_thread_loop, std::ref(workspace)) {}
} // namespace pc::networking