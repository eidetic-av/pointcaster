#include "mqtt_client.h"

#include <atomic>
#include <chrono>
#include <concepts>
#include <format>
#include <logger/logger.h>
#include <memory>
#include <mqtt/client.h>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <type_traits>
#include <uuid/uuid.h>
#include <workspace/workspace.h>
#include <workspace/workspace_socket.h>

namespace pc::publishers {

using namespace std::chrono;
using namespace std::chrono_literals;

namespace {

void mqtt_client_thread_worker(std::stop_token stop_token,
                               Workspace &workspace) {
  auto socket = WorkspaceSocket::create_subscriber();

  bool enabled;
  std::string broker_uri;
  std::string client_id;
  bool auto_reconnect;
  mqtt::connect_options mqtt_options;

  const auto sync_config_vars = [&] {
    std::scoped_lock lock(workspace.config_access);
    const auto &config = workspace.config.publishers.value().mqtt.value();
    enabled = config.enabled.value();
    broker_uri = config.broker_uri.value();
    client_id = config.client_id.value();
    auto_reconnect = config.auto_reconnect.value();
    // TODO any mqtt_options to fill in
  };

  sync_config_vars();

  auto client = std::make_unique<mqtt::client>(broker_uri, client_id);

  try {
    client->connect(mqtt_options);
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
          client->connect(mqtt_options);
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
          mqtt::message_ptr msg;
          using VariantType = std::decay_t<decltype(v)>;
          if constexpr (std::is_convertible<VariantType, std::string>()) {
            msg = mqtt::make_message(path, v);
          } else if constexpr (std::is_arithmetic<VariantType>()) {
            msg = mqtt::make_message(path, std::to_string(v));
          } else {
            msg = mqtt::make_message(path, std::format("{}", v));
          }
          try {
            client->publish(msg);
          } catch (mqtt::exception e) {
            pc::logger()->error("MQTT publish failed with exception: {}",
                                e.get_message());
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