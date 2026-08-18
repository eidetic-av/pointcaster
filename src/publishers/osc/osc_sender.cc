#include "osc_sender.h"
#include "pointcaster/point_cloud.h"
#include "publishers/osc/osc_sender_config.h"

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <format>
#include <logger/logger.h>
#include <memory>
#include <mutex>
#include <optional>
#include <profiling/profiling_zone.h>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <variant>
#include <workspace/workspace.h>
#include <workspace/workspace_socket.h>

#include <lo/lo.h>
#define LO_USE_EXCEPTIONS
#include <lo/lo_cpp.h>

namespace pc::publishers {

using namespace std::chrono;
using namespace std::chrono_literals;
using namespace pc::profiling;

namespace {

using PositionEncoding = OscSenderConfiguration::PositionEncoding;

// config paths are already '/' delimited like osc addresses, they just don't
// carry the leading slash an address needs
std::string osc_address_from(std::string_view path) {
  if (path.starts_with('/')) return std::string(path);
  return std::format("/{}", path);
}

// main worker thread
void osc_sender_thread_worker(std::stop_token stop_token,
                              Workspace &workspace) {
  auto socket = WorkspaceSocket::create_subscriber();

  bool enabled;
  std::string host;
  int port;
  PositionEncoding position_encoding;
  int max_cloud_points;

  const auto sync_config_vars = [&] {
    std::scoped_lock lock(workspace.config_access);
    const auto &config = workspace.config.publishers.value().osc.value();
    enabled = config.enabled.value();
    host = config.host.value();
    port = config.port.value();
    position_encoding = config.position_encoding.value();
    max_cloud_points = std::max(1, config.max_cloud_points.value());
  };

  sync_config_vars();

  std::unique_ptr<lo::Address> client;

  const auto open_client = [&] {
    try {
      client = std::make_unique<lo::Address>(host, port);
      pc::logger()->info("OSC sender publishing to '{}:{}'", host, port);
    } catch (const lo::Error &) {
      client.reset();
      pc::logger()->error("OSC sender failed to resolve '{}:{}'", host, port);
    }
  };

  open_client();

  while (!stop_token.stop_requested()) {

    const auto last_host = host;
    const auto last_port = port;

    sync_config_vars();

    if (!enabled) {
      std::this_thread::sleep_for(500ms);
      continue;
    }

    if (!client || host != last_host || port != last_port) {
      open_client();
      if (!client) {
        std::this_thread::sleep_for(2s);
        continue;
      }
    }

    const auto msg = socket.receive();
    if (msg == std::nullopt) continue;

    auto &[path, value] = msg.value();

    const auto send = [&](const std::string &address, const lo::Message &out) {
      ProfilingZone osc_send_zone("OscSender::send");
      try {
        if (client->send(address, out) == -1) {
          pc::logger()->error("OSC send to '{}' failed: {}", address,
                              client->errstr());
        }
      } catch (const lo::Error &) {
        pc::logger()->error("OSC send to '{}' failed with a liblo error",
                            address);
      } catch (const lo::Invalid &) {
        pc::logger()->error("OSC send to '{}' used an invalid liblo object",
                            address);
      }
    };

    // positions can go out as either raw millimeter ints or floats as metres
    const auto push_back_position = [&](lo::Message &out, const position &p) {
      if (position_encoding == PositionEncoding::Metres) {
        out.add_float(length{p.x}.metres());
        out.add_float(length{p.y}.metres());
        out.add_float(length{p.z}.metres());
      } else {
        out.add_int32(p.x);
        out.add_int32(p.y);
        out.add_int32(p.z);
      }
    };

    // OSC requires a point cloud to be sent as
    // individual elements, it has no array type:
    const auto send_cloud_elements =
        [&](std::string_view base, std::size_t count, auto &&fill_message) {
          const auto address = osc_address_from(base);
          const auto sent_count =
              std::min(count, static_cast<std::size_t>(max_cloud_points));

          {
            lo::Message out;
            out.add_int32(static_cast<int32_t>(sent_count));
            send(std::format("{}/count", address), out);
          }

          for (std::size_t i = 0; i < sent_count; i++) {
            lo::Message out;
            fill_message(out, i);
            send(std::format("{}/{}", address, i), out);
          }

          if (sent_count < count) {
            pc::logger()->debug(
                "OSC truncated '{}' to {} of {} points (max_cloud_points)",
                address, sent_count, count);
          }
        };

    try {
      std::visit(
          [&](auto &v) {
            ProfilingZone osc_zone("OscSender::process_msg");

            using VariantType = std::decay_t<decltype(v)>;

            if constexpr (std::same_as<VariantType, AabbListPtr>) {
              if (!v) return;
              // each aabb is one message of its min followed by its max
              send_cloud_elements(
                  path, v->min_positions().size(),
                  [&](lo::Message &out, std::size_t i) {
                    push_back_position(out, v->min_positions()[i]);
                    push_back_position(out, v->max_positions()[i]);
                  });
            } else if constexpr (pc::is_cloud_stream_v<VariantType>) {
              if (!v) return;
              send_cloud_elements(path, v->positions.size(),
                                  [&](lo::Message &out, std::size_t i) {
                                    push_back_position(out, v->positions[i]);
                                  });
            } else if constexpr (std::same_as<VariantType, position_bounds>) {
              // a bounds is one box: its min followed by its max, the same
              // message a single element of an aabb list sends
              lo::Message out;
              {
                ProfilingZone osc_encode_zone("OscSender::encode");
                push_back_position(out, v.min);
                push_back_position(out, v.max);
              }
              send(osc_address_from(path), out);
            } else {
              lo::Message out;
              {
                ProfilingZone osc_encode_zone("OscSender::encode");
                if constexpr (std::same_as<VariantType, bool>) {
                  out.add_bool(v);
                } else if constexpr (std::same_as<VariantType, radius> ||
                                     std::same_as<VariantType, length>) {
                  out.add_int32(static_cast<int32_t>(v.mm));
                } else if constexpr (std::same_as<VariantType, std::string>) {
                  out.add_string(v);
                } else if constexpr (std::same_as<VariantType, float>) {
                  out.add_float(v);
                } else if constexpr (std::same_as<VariantType, double>) {
                  out.add_double(v);
                } else if constexpr (std::is_integral_v<VariantType>) {
                  out.add_int32(static_cast<int32_t>(v));
                } else {
                  out.add_string(std::format("{}", v));
                }
              }
              send(osc_address_from(path), out);
            }
          },
          value);
    } catch (const lo::Error &) {
      pc::logger()->error("OSC failed to build a message for '{}'", path);
    } catch (const lo::Invalid &) {
      pc::logger()->error("OSC failed to build a message for '{}'", path);
    }
  }
}

} // namespace

OscSender::OscSender(Workspace &workspace)
    : _worker(osc_sender_thread_worker, std::ref(workspace)) {}

} // namespace pc::publishers
