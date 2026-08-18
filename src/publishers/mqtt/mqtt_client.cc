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

// some intermediary structs needed to serialize complex types over mqtt
struct StructuredAabb {
  std::array<int16_t, 3> min;
  std::array<int16_t, 3> max;
  MSGPACK_DEFINE_MAP(min, max);
  const std::string id;
};
struct StructuredAabbList {
  std::vector<StructuredAabb> aabbs;
  MSGPACK_DEFINE_ARRAY(aabbs);
};

// main worker thread
void mqtt_client_thread_worker(std::stop_token stop_token,
                               Workspace &workspace) {

  using EmptyMessageHandling = MqttClientConfiguration::EmptyMessageHandling;
  using SerializationFormat = MqttClientConfiguration::SerializationFormat;

  bool enabled;
  std::string broker_uri;
  std::string client_id;
  bool auto_reconnect;
  SerializationFormat serialization_format;
  bool serialize_as_structures;
  bool send_retained;
  EmptyMessageHandling empty_message_handling;

  const auto sync_config_vars = [&] {
    std::scoped_lock lock(workspace.config_access);
    const auto &config = workspace.config.publishers.value().mqtt.value();
    enabled = config.enabled.value();
    broker_uri = config.broker_uri.value();
    client_id = config.client_id.value();
    auto_reconnect = config.auto_reconnect.value();
    serialization_format = config.serialization_format.value();
    serialize_as_structures = config.serialize_as_structures.value();
    send_retained = config.send_retained.value();
    empty_message_handling = config.empty_message_handling.value();
  };

  sync_config_vars();

  // workspace input
  auto socket = WorkspaceSocket::create_subscriber();

  // broker output
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
        [&](auto &value) {
          ProfilingZone mqtt_zone("MqttClient::process_msg");

          using VariantType = std::decay_t<decltype(value)>;
          mqtt::message_ptr msg;
          bool payload_empty;
          {
            ProfilingZone mqtt_serialization_zone("MqttClient::serialize");

            if constexpr (std::is_convertible<VariantType, std::string>()) {
              msg = mqtt::make_message(path, value);
              payload_empty = std::empty(value);
            } else if constexpr (std::is_arithmetic<VariantType>()) {
              msg = mqtt::make_message(path, std::to_string(value));
              payload_empty = false;
            } else if constexpr (std::same_as<VariantType, AabbListPtr>) {
              if (!value) return;

              // we either serialize as typed aabb structs, which are easier to
              // read for instance in mqtt explorer as json, and may be easier
              // in some applications to decode, or we serialize as basic
              // array / list types of min/max bounds, which is more efficient

              if (serialize_as_structures) {
                const auto &min = value->min_positions();
                const auto &max = value->max_positions();
                const auto indices =
                    std::ranges::iota_view(0, static_cast<int>(value->size()));
                // TODO cpp26...
                // const auto indices =
                // std::views::indices(std::ranges::size(min));
                StructuredAabbList list{
                    .aabbs = std::views::zip(min, max, indices) |
                             std::views::transform([](const auto &aabb) {
                               const auto &[min, max, i] = aabb;
                               return StructuredAabb{
                                   .min = {min.x, min.y, min.z},
                                   .max = {max.x, max.y, max.z},
                                   // TODO id's are just indices for now
                                   .id = std::format("{}", i)};
                             }) |
                             std::ranges::to<std::vector>()};
                payload_empty = list.aabbs.empty();

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
                    std::views::zip(value->min_positions(),
                                    value->max_positions()) |
                    std::views::transform([](const auto &min_max) {
                      const auto &[min, max] = min_max;
                      return std_aabb{
                          {{min.x, min.y, min.z}, {max.x, max.y, max.z}}};
                    }) |
                    std::ranges::to<std::vector>();
                payload_empty = aabbs.empty();

                if (serialization_format == SerializationFormat::JSON) {
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
              if (!value) return;

              using std_position = std::array<int16_t, 3>;
              const auto positions = value->positions |
                                     std::views::transform([](const auto &p) {
                                       return std_position{{p.x, p.y, p.z}};
                                     }) |
                                     std::ranges::to<std::vector>();
              payload_empty = positions.empty();

              if (serialization_format == SerializationFormat::JSON) {
                ProfilingZone json_zone("serialize::json");
                msg = mqtt::make_message(path, rfl::json::write(positions));
              } else { // SerializationFormat::MessagePack
                ProfilingZone msgpack_zone("serialize::msgpack");
                msgpack::sbuffer send_buffer;
                msgpack::pack(send_buffer, positions);
                msg = mqtt::make_message(path, send_buffer.data(),
                                         send_buffer.size());
              }
            } else if constexpr (std::same_as<VariantType, position_bounds>) {
              // a bounds is one box, so it goes out in the shape a single
              // element of an aabb list would
              const std::array<int16_t, 3> min{value.min.x, value.min.y,
                                               value.min.z};
              const std::array<int16_t, 3> max{value.max.x, value.max.y,
                                               value.max.z};
              payload_empty = false;

              if (serialize_as_structures) {
                const StructuredAabb bounds{
                    .min = min, .max = max, .id = "bounds"};
                if (serialization_format == SerializationFormat::JSON) {
                  ProfilingZone json_zone("serialize::json");
                  msg = mqtt::make_message(path, rfl::json::write(bounds));
                } else { // SerializationFormat::MessagePack
                  ProfilingZone msgpack_zone("serialize::msgpack");
                  msgpack::sbuffer send_buffer;
                  msgpack::pack(send_buffer, bounds);
                  msg = mqtt::make_message(path, send_buffer.data(),
                                           send_buffer.size());
                }
              } else {
                const std::array<std::array<int16_t, 3>, 2> bounds{min, max};
                if (serialization_format == SerializationFormat::JSON) {
                  ProfilingZone json_zone("serialize::json");
                  msg = mqtt::make_message(path, rfl::json::write(bounds));
                } else { // SerializationFormat::MessagePack
                  ProfilingZone msgpack_zone("serialize::msgpack");
                  msgpack::sbuffer send_buffer;
                  msgpack::pack(send_buffer, bounds);
                  msg = mqtt::make_message(path, send_buffer.data(),
                                           send_buffer.size());
                }
              }
            } else {
              const auto msg_str = std::format("{}", value);
              payload_empty = msg_str.empty();
              msg = mqtt::make_message(path, std::move(msg_str));
            }
          }
          {
            ProfilingZone mqtt_publish_zone("MqttClient::publish");

            static thread_local bool has_published_empty_once = false;
            msg->set_retained(send_retained);

            try {
              if (!payload_empty) {
                client->publish(msg);
                has_published_empty_once = false;
              } else if (empty_message_handling ==
                         EmptyMessageHandling::PublishEmptyAlways) {
                client->publish(msg);
              } else if (empty_message_handling ==
                             EmptyMessageHandling::PublishEmptyOnce &&
                         !has_published_empty_once) {
                client->publish(msg);
                has_published_empty_once = true;
              }
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