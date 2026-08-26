#pragma once

#include "../workspace_listener.h"
#include "mqtt_client_config.h"
#include <memory>

namespace pc::publishers {

class MqttConnection;

class MqttClient {
public:
  MqttClient(Workspace &workspace);
  ~MqttClient();

  MqttClient(const MqttClient &) = delete;
  MqttClient &operator=(const MqttClient &) = delete;
  MqttClient(MqttClient &&) = delete;
  MqttClient &operator=(MqttClient &&) = delete;

  MqttClientConfiguration config(Workspace &workspace) const;

  void tick();

  void handle_update(const std::string_view path, const ConfigValue &value,
                     const MqttClientConfiguration &config_snapshot) const;

  void handle_config_change(std::string_view path,
                            const MqttClientConfiguration &config_snapshot);

private:
  std::unique_ptr<MqttConnection> _connection;
  bool _auto_reconnect = false;

  // must stay last
  WorkspaceListener<MqttClient, MqttClientConfiguration> _listener;
};
} // namespace pc::publishers
