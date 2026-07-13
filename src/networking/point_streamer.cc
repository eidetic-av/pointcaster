#include "point_streamer.h"

#include <algorithm>
#include <chrono>
#include <core/logger/logger.h>
#include <core/profiling/profiling_zone.h>
#include <cstddef>
#include <cstring>
#include <format>
#include <memory>
#include <mutex>
#include <networking/stream_channels.h>
#include <session/session.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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

namespace pc::networking {
namespace {
using namespace std::chrono;
using namespace pc::profiling;

constexpr auto period_for(int hz) {
  return duration_cast<steady_clock::duration>(
      duration<double>(1.0 / std::max(hz, 1)));
}

bool channel_enabled(
    const std::unordered_map<std::string, bool> &enabled_overrides,
    const std::string &address) {
  const auto it = enabled_overrides.find(address);
  return it == enabled_overrides.end() || it->second;
}

void streaming_thread_loop(
    std::stop_token stop_token, Workspace &workspace,
    std::atomic<std::shared_ptr<const std::unordered_map<std::string, int>>>
        &subscriber_counts_out) {
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

  zmq::socket_t pub_socket{ctx, zmq::socket_type::xpub};
  pub_socket.set(zmq::sockopt::sndhwm, 32);
  pub_socket.set(zmq::sockopt::linger, 0);
  pub_socket.set(zmq::sockopt::xpub_verboser, 1);

  try {
    pub_socket.bind(std::format("tcp://{}:{}", address, port));
  } catch (const zmq::error_t &e) {
    pc::logger()->error("Point streamer failed to bind tcp://{}:{} - {}",
                        address, port, e.what());
    return;
  }
  pc::logger()->info("Point streamer bound to tcp://{}:{}", address, port);

  // per-channel cached state, addressed by channel address
  // the clouds themselves
  std::unordered_map<std::string, std::shared_ptr<PointCloud>> last_clouds;
  // pre-serialized data: address + '\0' + serialized payload
  std::unordered_map<std::string, std::shared_ptr<std::vector<std::byte>>>
      last_data;

  std::unordered_map<std::string, int> subscriber_counts;

  auto next_tick = steady_clock::now();

  while (!stop_token.stop_requested()) {
    // drain subscribe/unsubscribe notifications without blocking
    while (true) {
      zmq::message_t msg;
      const auto result = pub_socket.recv(msg, zmq::recv_flags::dontwait);
      if (!result) break;
      if (msg.size() < 1) continue;

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
    }

    int publish_hz = 30;
    bool compress = false;
    std::vector<StreamChannelSource> sources;

    {
      std::lock_guard lock(workspace.config_access);
      const auto &stream_config = workspace.config.point_streamer.value();
      publish_hz = stream_config.publish_hz.value();
      compress = stream_config.compress.value();

      std::unordered_map<std::string, bool> enabled_overrides;
      for (const auto &channel : stream_config.channels)
        enabled_overrides[channel.address] = channel.enabled.value();

      auto all_sources = collect_stream_channel_sources(workspace);
      sources.reserve(all_sources.size());
      for (auto &source : all_sources) {
        const auto it = subscriber_counts.find(source.address);
        if (it == subscriber_counts.end()) continue;
        const auto has_subscriber = it->second > 0;
        if (has_subscriber &&
            channel_enabled(enabled_overrides, source.address)) {
          sources.push_back(std::move(source));
        }
      }
    }

    // drop cached state for channels that no longer exist
    {
      std::unordered_set<std::string> live_addresses;
      live_addresses.reserve(sources.size());
      for (const auto &source : sources) live_addresses.insert(source.address);

      std::erase_if(last_clouds, [&](const auto &kv) {
        return !live_addresses.contains(kv.first);
      });
      std::erase_if(last_data, [&](const auto &kv) {
        return !live_addresses.contains(kv.first);
      });
    }

    // find channels whose point cloud changed and needs re-serializing
    std::vector<size_t> to_serialize;
    for (size_t i = 0; i < sources.size(); ++i) {
      const auto &source = sources[i];
      if (!source.cloud || source.cloud->empty()) continue;
      const auto it = last_clouds.find(source.address);
      if (it == last_clouds.end() || it->second != source.cloud)
        to_serialize.push_back(i);
    }

    // serialize all changed channels in parallel, then merge results
    // sequentially before any sending happens
    if (!to_serialize.empty()) {
      ProfilingZone serialize_zone("point_stream::serialize");

      std::vector<std::shared_ptr<std::vector<std::byte>>> serialized(
          to_serialize.size());
      {
        ProfilingZone parallel_serialize_zone("parallel_serailize");
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, to_serialize.size()),
            [&](const tbb::blocked_range<size_t> &range) {
              for (size_t r = range.begin(); r < range.end(); ++r) {
                const auto &source = sources[to_serialize[r]];
                serialized[r] = std::make_shared<std::vector<std::byte>>(
                    source.cloud->serialize(compress));
              }
            });
      }
      {
        ProfilingZone parallel_serialize_zone("mempcy_frame");
        for (size_t r = 0; r < to_serialize.size(); ++r) {
          const auto &source = sources[to_serialize[r]];
          const auto &payload = *serialized[r];

          auto framed = std::make_shared<std::vector<std::byte>>(
              source.address.size() + 1 + payload.size());
          std::memcpy(framed->data(), source.address.data(),
                      source.address.size());
          (*framed)[source.address.size()] = std::byte{0};
          std::memcpy(framed->data() + source.address.size() + 1,
                      payload.data(), payload.size());

          last_clouds[source.address] = source.cloud;
          last_data[source.address] = std::move(framed);
        }
      }
    }

    // TODO do we actually want a re-send? make this a param in a per-channel
    // config maybe... re-send every enabled channel's last known frame. XPUB
    // drops sends with no subscribers, so a late-joining subscriber would
    // otherwise never get a frame until the next change.
    {
      ProfilingZone send_zone("point_stream::send");
      for (const auto &source : sources) {
        const auto it = last_data.find(source.address);
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
                             source.address, e.what());
        }
      }
    }

    subscriber_counts_out.store(
        std::make_shared<const std::unordered_map<std::string, int>>(
            subscriber_counts),
        std::memory_order_release);

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
  const auto it = counts->find(channel_address);
  return it != counts->end() && it->second > 0;
}

} // namespace pc::networking
