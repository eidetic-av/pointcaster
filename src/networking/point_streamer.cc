#include "point_streamer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef ZMQ_BUILD_DRAFT_API
#define ZMQ_BUILD_DRAFT_API
#endif
#include <zmq.hpp>
#include <zpp_bits.h>

#include <core/logger/logger.h>
#include <core/profiling/profiling_zone.h>
#include <session/session.h>
#include <workspace/workspace.h>
#include <workspace/workspace_config.h>

namespace pc::networking {
namespace {
using namespace std::chrono;
using namespace pc::profiling;

constexpr auto period_for(int hz) {
  return duration_cast<steady_clock::duration>(
      duration<double>(1.0 / std::max(hz, 1)));
}

void streaming_thread_loop(std::stop_token stop_token, Workspace &workspace) {
  std::string address;
  int port = 9992;
  {
    std::lock_guard lock(workspace.config_access);
    const auto &config = workspace.config.point_streamer.value();
    address = config.address.value();
    port = config.port.value();
  }
  if (address.empty()) address = "*";

  zmq::context_t ctx{1};

  zmq::socket_t live_socket{ctx, zmq::socket_type::radio};
  live_socket.set(zmq::sockopt::sndhwm, 1);
  live_socket.set(zmq::sockopt::linger, 0);
  live_socket.set(zmq::sockopt::conflate, 1);

  try {
    live_socket.bind(std::format("tcp://{}:{}", address, port));
  } catch (const zmq::error_t &e) {
    pc::logger()->error("Point streamer failed to bind tcp://{}:{} - {}",
                        address, port, e.what());
    return;
  }
  pc::logger()->info("Point streamer live=tcp://{}:{}", address, port);

  std::shared_ptr<PointCloud> last_live_cloud;
  std::shared_ptr<std::vector<std::byte>> live_buf; // cached serialised frame
  auto next_tick = steady_clock::now();

  while (!stop_token.stop_requested()) {
    int publish_hz = 30;
    bool compress = false;
    std::shared_ptr<PointCloud> live_cloud;

    {
      std::lock_guard lock(workspace.config_access);
      const auto &stream_config = workspace.config.point_streamer.value();
      publish_hz = stream_config.publish_hz.value();
      compress = stream_config.compress.value();
      for (auto &[id, session] : workspace.sessions) {
        if (!session) continue;
        live_cloud = session->point_cloud();
        break;
      }
    }

    // Re-serialise only when the cloud actually changes...
    if (live_cloud && !live_cloud->empty() && live_cloud != last_live_cloud) {
      ProfilingZone serailize_zone("point_stream::serialize");
      last_live_cloud = live_cloud;
      std::vector<std::byte> serialized_cloud_buffer;
      {
        ProfilingZone z("point_stream::serialize");
        // serialized_cloud_buffer = live_cloud->serialize(compress);
        serialized_cloud_buffer = live_cloud->serialize(false);
      }
      {
        ProfilingZone z("point_stream::make_shared");
        live_buf = std::make_shared<std::vector<std::byte>>(
            std::move(serialized_cloud_buffer));
      }
    }

    // ...but re-send every tick. RADIO drops sends with no connected peer, so a
    // DISH that joins after the last change would otherwise never get a frame.
    if (live_buf && !live_buf->empty()) {
      ProfilingZone serailize_zone("point_stream::send");

      // hand libzmq the bytes (no copy), keep them alive via a
      // heap shared_ptr freed by the deleter once the send completes
      auto *hint = new std::shared_ptr<std::vector<std::byte>>(live_buf);
      zmq::message_t msg(
          live_buf->data(), live_buf->size(),
          [](void *, void *h) {
            delete static_cast<std::shared_ptr<std::vector<std::byte>> *>(h);
          },
          hint);
      try {
        msg.set_group("live");
        live_socket.send(msg, zmq::send_flags::none);
      } catch (const zmq::error_t &e) {
        pc::logger()->warn("Point streamer send failed on 'live': {}",
                           e.what());
      }
    }

    next_tick =
        std::max(next_tick + period_for(publish_hz), steady_clock::now());
    std::this_thread::sleep_until(next_tick);
  }
}
} // namespace

PointStreamer::PointStreamer(Workspace &workspace)
    : _worker(streaming_thread_loop, std::ref(workspace)) {}

} // namespace pc::networking