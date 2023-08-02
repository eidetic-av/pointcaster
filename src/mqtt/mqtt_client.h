#pragma once

#include "mqtt_client_config.h"
#include <expected>
#include <future>
#include <memory>
#include <mqtt/async_client.h>
#include <string_view>
#include <mutex>

namespace pc {

class MqttClient {
public:

  static std::shared_ptr<MqttClient> &instance() {
    if (!_instance)
      throw std::logic_error(
	  "No MqttClient instance created. Call MqttClient::create() first.");
    return _instance;
  }

  static void create(MqttClientConfiguration &config) {
    std::lock_guard lock(_mutex);
    _instance = make_instance(config);
  }

  MqttClient(const MqttClient &) = delete;
  MqttClient &operator=(const MqttClient &) = delete;

  ~MqttClient();

  void connect();
  void connect(const std::string_view uri);
  void connect(const std::string_view host_name, const int port);

  void disconnect();

  void set_uri(const std::string_view uri);
  void set_uri(const std::string_view hostname, int port);

  void publish(const std::string_view topic, const std::string_view message);
  void publish(const std::string_view topic, const std::vector<uint8_t> &bytes);

  void draw_imgui_window();

private:
  static std::mutex _mutex;
  static std::shared_ptr<MqttClient> _instance;

  static std::shared_ptr<MqttClient>
  make_instance(MqttClientConfiguration &config) {
    return std::shared_ptr<MqttClient>(new MqttClient(config));
  }

  explicit MqttClient(MqttClientConfiguration &config);

  MqttClientConfiguration &_config;
  std::future<void> _connection_future;
  std::unique_ptr<mqtt::async_client> _client;

  bool check_connected();
};

} // namespace pc
