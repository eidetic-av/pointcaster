#include "mqtt_client.h"
#include "pointcaster/point_cloud.h"
#include "publishers/mqtt/mqtt_client_config.h"

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <format>
#include <logger/logger.h>
#include <memory>
#include <mqtt/client.h>
#include <msgpack.hpp>
#include <mutex>
#include <optional>
#include <profiling/profiling_zone.h>
#include <ranges>
#include <rfl/json.hpp>
#include <stop_token>
#include <string>
#include <type_traits>
#include <uuid/uuid.h>
#include <variant>
#include <vector>
#include <workspace/workspace.h>
#include <workspace/workspace_socket.h>

namespace pc::publishers {

using namespace std::chrono;
using namespace std::chrono_literals;
using namespace pc::profiling;

namespace {

// some intermediary structs needed to send complex types over mqtt
struct StructuredAabb {
  std::array<int16_t, 3> min;
  std::array<int16_t, 3> max;
  MSGPACK_DEFINE_MAP(min, max);
};
struct StructuredAabbList {
  std::vector<StructuredAabb> aabbs;
  MSGPACK_DEFINE_ARRAY(aabbs);
};

// main worker thread
void mqtt_client_thread_worker(std::stop_token stop_token,
                               Workspace &workspace) {
  auto socket = WorkspaceSocket::create_subscriber();

  bool enabled;
  std::string broker_uri;
  std::string client_id;
  bool auto_reconnect;
  MqttClientConfiguration::SerializationFormat serialization_format;
  bool serialize_as_structures;

  const auto sync_config_vars = [&] {
    std::scoped_lock lock(workspace.config_access);
    const auto &config = workspace.config.publishers.value().mqtt.value();
    enabled = config.enabled.value();
    broker_uri = config.broker_uri.value();
    client_id = config.client_id.value();
    auto_reconnect = config.auto_reconnect.value();
    serialization_format = config.serialization_format.value();
    serialize_as_structures = config.serialize_as_structures.value();
  };

  sync_config_vars();

  auto client = std::make_unique<mqtt::client>(broker_uri, client_id);

  try {
    client->connect();
    pc::logger()->info("MQTT client connected to '{}' as '{}'", broker_uri,
                       client_id);
  } catch (mqtt::exception e) {
    pc::logger()->error("MQTT connect exception: {}", e.get_message());
  }

  while (!stop_token.stop_requested()) {

    const auto last_broker_uri = broker_uri;
    const auto last_client_id = client_id;

    sync_config_vars();

    if (!enabled) {
      std::this_thread::sleep_for(500ms);
      if (client->is_connected()) client->disconnect();
      continue;
    }

    bool force_reconnect = false;
    if (client->is_connected() &&
        (broker_uri != last_broker_uri || client_id != last_client_id)) {
      client->disconnect();
      force_reconnect = true;
    }

    if (!client->is_connected()) {
      std::this_thread::sleep_for(2s);
      sync_config_vars();
      if (auto_reconnect || force_reconnect) {
        try {
          client = std::make_unique<mqtt::client>(broker_uri, client_id);
          client->connect();
          pc::logger()->info("MQTT client connected to '{}' as '{}'",
                             broker_uri, client_id);
        } catch (mqtt::exception e) {
          pc::logger()->error("MQTT auto-reconnect failed with exeception: {}",
                              e.get_message());
          continue;
        }
      } else {
        continue;
      }
    }

    const auto msg = socket.receive();
    if (msg == std::nullopt) continue;

    auto &[path, value] = msg.value();

    std::visit(
        [&](auto &v) {
          ProfilingZone mqtt_zone("MqttClient::process_msg");

          using VariantType = std::decay_t<decltype(v)>;
          mqtt::message_ptr msg;
          {
            ProfilingZone mqtt_serialization_zone("MqttClient::serialize");

            if constexpr (std::is_convertible<VariantType, std::string>()) {
              msg = mqtt::make_message(path, v);
            } else if constexpr (std::is_arithmetic<VariantType>()) {
              msg = mqtt::make_message(path, std::to_string(v));
            } else if constexpr (std::same_as<VariantType, AabbListPtr>) {
              if (!v) return;

              // we either serialize as typed aabb structs, which are easier to
              // read for instance in mqtt explorer as json, and may be easier
              // in some applications to decode, or we serialize as basic
              // array / list types of min/max bounds, which is more efficient

              if (serialize_as_structures) {
                StructuredAabbList list{
                    .aabbs =
                        std::views::zip(v->min_positions(),
                                        v->max_positions()) |
                        std::views::transform([](const auto &min_max) {
                          const auto &[min, max] = min_max;
                          return StructuredAabb{.min = {min.x, min.y, min.z},
                                                .max = {max.x, max.y, max.z}};
                        }) |
                        std::ranges::to<std::vector>()};

                if (serialization_format ==
                    MqttClientConfiguration::SerializationFormat::JSON) {
                  ProfilingZone json_zone("serialize::json");
                  msg = mqtt::make_message(path, rfl::json::write(list));
                } else { // SerializationFormat::MessagePack
                  ProfilingZone msgpack_zone("serialize::msgpack");
                  msgpack::sbuffer send_buffer;
                  msgpack::pack(send_buffer, list);
                  msg = mqtt::make_message(path, send_buffer.data(),
                                           send_buffer.size());
                }
              } else {
                using std_aabb = std::array<std::array<int16_t, 3>, 2>;
                const auto aabbs =
                    std::views::zip(v->min_positions(), v->max_positions()) |
                    std::views::transform([](const auto &min_max) {
                      const auto &[min, max] = min_max;
                      return std_aabb{
                          {{min.x, min.y, min.z}, {max.x, max.y, max.z}}};
                    }) |
                    std::ranges::to<std::vector>();

                if (serialization_format ==
                    MqttClientConfiguration::SerializationFormat::JSON) {
                  ProfilingZone json_zone("serialize::json");
                  msg = mqtt::make_message(path, rfl::json::write(aabbs));
                } else { // SerializationFormat::MessagePack
                  ProfilingZone msgpack_zone("serialize::msgpack");
                  msgpack::sbuffer send_buffer;
                  msgpack::pack(send_buffer, aabbs);
                  msg = mqtt::make_message(path, send_buffer.data(),
                                           send_buffer.size());
                }
              }
            } else if constexpr (pc::is_cloud_stream_v<VariantType>) {
              if (!v) return;

              using std_position = std::array<int16_t, 3>;
              const auto positions =
                  v->positions | std::views::transform([](const auto &p) {
                    return std_position{{p.x, p.y, p.z}};
                  }) |
                  std::ranges::to<std::vector>();

              if (serialization_format ==
                  MqttClientConfiguration::SerializationFormat::JSON) {
                ProfilingZone json_zone("serialize::json");
                msg = mqtt::make_message(path, rfl::json::write(positions));
              } else { // SerializationFormat::MessagePack
                ProfilingZone msgpack_zone("serialize::msgpack");
                msgpack::sbuffer send_buffer;
                msgpack::pack(send_buffer, positions);
                msg = mqtt::make_message(path, send_buffer.data(),
                                         send_buffer.size());
              }
            } else {
              msg = mqtt::make_message(path, std::format("{}", v));
            }
          }
          {
            ProfilingZone mqtt_publish_zone("MqttClient::publish");
            try {
              client->publish(msg);
            } catch (mqtt::exception e) {
              pc::logger()->error("MQTT publish failed with exception: {}",
                                  e.get_message());
            }
          }
        },
        value);
  }

  if (client && client->is_connected()) client->disconnect();
}

} // namespace

MqttClient::MqttClient(Workspace &workspace)
    : _worker(mqtt_client_thread_worker, std::ref(workspace)) {}

} // namespace pc::publishers