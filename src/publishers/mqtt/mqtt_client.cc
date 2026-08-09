#include "mqtt_client.h"

#include <chrono>
#include <logger/logger.h>
#include <mutex>
#include <optional>
#include <stop_token>
#include <workspace/workspace.h>
#include <workspace/workspace_socket.h>

namespace pc::publishers {

using namespace std::chrono;
using namespace std::chrono_literals;

namespace {

void mqtt_client_thread_worker(std::stop_token stop_token,
                               Workspace &workspace) {

  auto &config = workspace.config.publishers.value().mqtt.value();
  std::string broker_uri;
  {
    std::scoped_lock lock(workspace.config_access);
    broker_uri = config.broker_uri.value();
  }

  auto socket = WorkspaceSocket::create_subscriber("");

  // create client
  pc::logger()->debug("Create mqtt client at '{}'", broker_uri);

  while (!stop_token.stop_requested()) {

    bool enabled;
    {
      std::scoped_lock lock(workspace.config_access);
      enabled = config.enabled.value();
    }
    if (!enabled) {
      std::this_thread::sleep_for(500ms);
      continue;
    }

    const auto msg = socket.receive();
    if (msg == std::nullopt) continue;

    auto &[path, value] = msg.value();
    std::visit(
        [&](auto &v) {
          // do mqtt publisher path here
          pc::logger()->debug("mqtt -> {}: {}", path, v);
        },
        value);
  }
}

} // namespace

MqttClient::MqttClient(Workspace &workspace)
    : _worker(mqtt_client_thread_worker, std::ref(workspace)) {}

} // namespace pc::publishers