#pragma once
#include <mqtt/async_client.h>
#include <string_view>

namespace pc {

class MqttClient {
public:
  MqttClient(std::string_view client_id, std::string_view uri);
  ~MqttClient();

  void connect();
  void disconnect();
  void publish(std::string_view topic, std::string_view message);

private:
  std::unique_ptr<mqtt::async_client> _client;
  std::string_view _client_id;
  std::string_view _uri;
};

} // namespace pc::mqtt
