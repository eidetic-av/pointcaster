#include "point_streamer.h"
#include "networking/zmq_context.h"
#include "point_streamer/point_streamer_config.h"

#include <algorithm>
#include <chrono>
#include <core/logger/logger.h>
#include <core/profiling/profiling_zone.h>
#include <cstddef>
#include <cstring>
#include <format>
#include <memory>
#include <mutex>
#include <networking/zmq_context.h>
#include <point_streamer/stream_channels.h>
#include <ranges>
#include <session/session.h>
#include <span>
#include <string>
#include <thread>
#include <util/string_map.h>
#include <vector>
#include <workspace/workspace.h>
#include <workspace/workspace_config.h>
#include <zmq.hpp>
#include <zpp_bits.h>

// tbb hates qt emit macro i think...
#pragma push_macro("emit")
#undef emit
#include <oneapi/tbb/parallel_for.h>
#pragma pop_macro("emit")

using namespace std::chrono;
using namespace pc::profiling;

namespace pc::networking {

namespace {

constexpr auto period_for(int hz) {
  return duration_cast<steady_clock::duration>(
      duration<double>(1.0 / std::max(hz, 1)));
}

bool channel_enabled(
    const std::span<const StreamChannelConfiguration> channel_configs,
    const std::string_view address) {
  const auto it =
      std::find_if(channel_configs.begin(), channel_configs.end(),
                   [&](auto &config) { return config.address == address; });
  if (it == channel_configs.end()) return true;
  return it->enabled.value();
}

void streaming_thread_loop(
    std::stop_token stop_token, Workspace &workspace,
    std::atomic<std::shared_ptr<const StringMap<int>>> &subscriber_counts_out) {

  std::string address;
  int port;
  int publish_hz = 30;
  bool compress = false;
  std::vector<PointStream> point_streams;
  std::vector<StreamChannelConfiguration> channel_configs;

  // syncing access... pattern here?
  // TODO this config sync with new local values could be made generic
  // somehow, ive used it over configs elsewhere / check mqtt_client.cc
  // ALSO the whole syncing strings and collections each frame seems wasteful
  const auto sync_config_vars = [&] {
    std::lock_guard lock(workspace.config_access);
    const auto &stream_config = workspace.config.point_streamer.value();
    address = stream_config.address.value();
    port = stream_config.port.value();
    publish_hz = stream_config.publish_hz.value();
    compress = stream_config.compress.value();
    // TODO is this too heavy to do every frame? maybe we need a dirty marker
    channel_configs = stream_config.channels;
    point_streams = collect_point_streams(workspace);
  };

  sync_config_vars();

  if (address.empty()) address = "*";

  auto &ctx = pc::networking::zmq_context();

  zmq::socket_t pub_socket{ctx, zmq::socket_type::xpub};
  pub_socket.set(zmq::sockopt::sndhwm, 32);
  pub_socket.set(zmq::sockopt::linger, 0);
  // adding the verboser with xpub ensures we get:
  // subscribe events, "unsubscribe" events for clients that drop off,
  pub_socket.set(zmq::sockopt::xpub_verboser, 1);

  try {
    pub_socket.bind(std::format("tcp://{}:{}", address, port));
  } catch (const zmq::error_t &e) {
    pc::logger()->error("Point streamer failed to bind tcp://{}:{} ({})",
                        address, port, e.what());
    return;
  } catch (...) {
    pc::logger()->error(
        "Point streamer failed to bind to {}:{} (Unknown exception)", address,
        port);
    return;
  }
  pc::logger()->info("Point streamer bound to tcp://{}:{}", address, port);

  // per-stream cached state, key is the streams address
  StringMap<std::shared_ptr<PointCloud>> last_clouds;
  // pre-serialized data: address + '\0' + serialized payload
  StringMap<std::shared_ptr<std::vector<std::byte>>> last_data;

  // an empty point cloud that is sent on the frame a channel transitions
  // from publishing to not publishing
  const auto stopped_cloud = std::make_shared<PointCloud>();

  StringMap<int> subscriber_counts;
  bool subscriber_counts_dirty = true;

  const auto handle_subscriber_message = [&](auto &msg) {
    if (msg.size() < 1) return;
    const auto *bytes = static_cast<const std::byte *>(msg.data());
    const bool subscribe = static_cast<unsigned char>(bytes[0]) == 1;
    std::string topic(reinterpret_cast<const char *>(bytes) + 1,
                      msg.size() - 1);
    if (!topic.empty() && topic.back() == '\0') topic.pop_back();
    auto &count = subscriber_counts[topic];
    if (subscribe) {
      count++;
    } else if (count > 0) {
      count--;
    }
    subscriber_counts_dirty = true;
  };

  const auto has_subscriber = [&](const std::string_view stream_address) {
    return std::ranges::any_of(subscriber_counts, [&](const auto &entry) {
      return entry.second > 0 && stream_address.starts_with(entry.first);
    });
  };

  // ---- processing loop ----

  auto next_tick = steady_clock::now();

  while (!stop_token.stop_requested()) {

    for (zmq::message_t msg; pub_socket.recv(msg, zmq::recv_flags::dontwait);) {
      handle_subscriber_message(msg);
    }

    sync_config_vars();

    std::vector<PointStream> publishing_streams;
    {
      ProfilingZone collect_streams_zone("point_stream::collect_streams");
      publishing_streams =
          point_streams | std::views::filter([&](const auto &stream) {
            return has_subscriber(stream.address) &&
                   channel_enabled(channel_configs, stream.address);
          }) |
          std::ranges::to<std::vector>();
    }

    // drop cached state for channels that no longer publish
    {
      const auto still_publishing = [&](const std::string &channel_address) {
        return std::ranges::any_of(publishing_streams, [&](const auto &stream) {
          return stream.address == channel_address;
        });
      };
      // erase the cache for no longer publishing channels and also
      // collect which ones that were stopped on this frame
      std::vector<PointStream> stopped_streams;
      std::erase_if(last_clouds, [&](const auto &entry) {
        if (still_publishing(entry.first)) return false;
        if (entry.second && !entry.second->empty()) {
          stopped_streams.push_back({entry.first, stopped_cloud});
        }
        return true;
      });
      std::erase_if(last_data, [&](const auto &entry) {
        return !still_publishing(entry.first);
      });
      // for each of the streams that stopped this frame, we send one last
      // pointcloud that just has zero points on its channel
      publishing_streams.insert(publishing_streams.end(),
                                stopped_streams.begin(), stopped_streams.end());
    }

    std::vector<PointStream> streams_to_serialize;
    {
      ProfilingZone collect_dirty_zone("point_stream::collect_dirty");
      streams_to_serialize =
          publishing_streams | std::views::filter([&](const auto &stream) {
            if (!stream.cloud) return false;
            if (stream.cloud->empty() && stream.cloud != stopped_cloud) {
              return false;
            }
            const auto it = last_clouds.find(stream.address);
            return it == last_clouds.end() || it->second != stream.cloud;
          }) |
          std::ranges::to<std::vector>();
    }

    // serialize all changed channels in parallel, then merge results
    // sequentially before any sending happens
    if (!streams_to_serialize.empty()) {

      ProfilingZone serialize_zone("point_stream::serialize");

      std::vector<std::shared_ptr<std::vector<std::byte>>> serialized_frames(
          streams_to_serialize.size());

      {
        ProfilingZone parallel_serialize_zone("parallel_serialize");
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, streams_to_serialize.size()),
            [&](const tbb::blocked_range<size_t> &range) {
              for (size_t i = range.begin(); i < range.end(); i++) {
                const auto &stream = streams_to_serialize[i];
                const auto payload =
                    stream.cloud->serialize(compress && !stream.cloud->empty());
                const auto prefix_size = stream.address.size() + 1;

                auto framed = std::make_shared<std::vector<std::byte>>(
                    prefix_size + payload.size());
                std::memcpy(framed->data(), stream.address.data(),
                            stream.address.size());
                (*framed)[stream.address.size()] = std::byte{0};
                std::memcpy(framed->data() + prefix_size, payload.data(),
                            payload.size());

                serialized_frames[i] = std::move(framed);
              }
            });
      }
      {
        ProfilingZone cache_frames_zone("cache_frames");
        for (size_t i = 0; i < streams_to_serialize.size(); i++) {
          const auto &stream = streams_to_serialize[i];
          last_clouds[stream.address] = stream.cloud;
          last_data[stream.address] = std::move(serialized_frames[i]);
        }
      }
    }

    // TODO do we actually want a re-send? make this a param in a per-channel
    // config maybe... re-send every enabled channel's last known frame. XPUB
    // drops sends with no subscribers, so a late-joining subscriber would
    // otherwise never get a frame until the next change.
    {
      ProfilingZone send_zone("point_stream::send");
      for (const auto &stream : publishing_streams) {
        const auto it = last_data.find(stream.address);
        if (it == last_data.end() || !it->second || it->second->empty())
          continue;

        // hand libzmq the bytes (no copy), keep them alive via a heap
        // shared_ptr freed by the deleter once the send completes
        auto *hint = new std::shared_ptr<std::vector<std::byte>>(it->second);
        zmq::message_t msg(
            it->second->data(), it->second->size(),
            [](void *, void *h) {
              delete static_cast<std::shared_ptr<std::vector<std::byte>> *>(h);
            },
            hint);
        try {
          pub_socket.send(msg, zmq::send_flags::none);
        } catch (const zmq::error_t &e) {
          pc::logger()->warn("Point streamer send failed on '{}': {}",
                             stream.address, e.what());
        }
      }
    }

    if (subscriber_counts_dirty) {
      subscriber_counts_out.store(
          std::make_shared<const StringMap<int>>(subscriber_counts),
          std::memory_order_release);
      subscriber_counts_dirty = false;
    }

    next_tick =
        std::max(next_tick + period_for(publish_hz), steady_clock::now());
    std::this_thread::sleep_until(next_tick);
  }
}
} // namespace

PointStreamer::PointStreamer(Workspace &workspace)
    : _worker(streaming_thread_loop, std::ref(workspace),
              std::ref(_subscriber_counts)) {}

bool PointStreamer::has_listeners(const std::string &channel_address) const {
  const auto counts = _subscriber_counts.load(std::memory_order_acquire);
  if (!counts) return false;
  const auto subscribe_all = counts->find("");
  if (subscribe_all != counts->end() && subscribe_all->second > 0) return true;
  const auto it = counts->find(channel_address);
  return it != counts->end() && it->second > 0;
}

std::shared_ptr<const StringMap<int>> PointStreamer::subscriber_counts() const {
  return _subscriber_counts.load(std::memory_order_acquire);
}

} // namespace pc::networking
