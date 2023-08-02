#pragma once

#include "mqtt_client_config.h"
#include <mqtt/async_client.h>
#include <string_view>
#include <future>
#include <expected>

namespace pc {

class MqttClient {
public:
  MqttClient(MqttClientConfiguration& config);
  ~MqttClient();

  void connect();
  void connect(const std::string_view uri);
  void connect(const std::string_view host_name, const int port);

  void disconnect();

  void publish(std::string_view topic, std::string_view message);

  void set_uri(const std::string_view uri);
  void set_uri(const std::string_view hostname, int port);

  void draw_imgui_window();

private:
  MqttClientConfiguration& _config;
  std::future<void> _connection_future;
  std::unique_ptr<mqtt::async_client> _client;
};

} // namespace pc::mqtt
