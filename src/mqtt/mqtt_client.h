#pragma once

#include "mqtt_client_config.h"
#include <atomic>
#include <expected>
#include <future>
#include <memory>
#include <mqtt/async_client.h>
#include <mutex>
#include <string_view>
#include <msgpack.hpp>

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
    std::call_once(_instantiated, [&config]() {
      _instance = std::shared_ptr<MqttClient>(new MqttClient(config));
    });
  }

  static bool connected() { return _connected; };

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
  void publish(const std::string_view topic, const void *payload,
	       const std::size_t size);

  template <typename T>
  void publish(const std::string_view topic, const T &data) {
    msgpack::sbuffer send_buffer;
    msgpack::pack(send_buffer, data);
    publish(topic, send_buffer.data(), send_buffer.size());
  }

  void draw_imgui_window();

private:
  static std::shared_ptr<MqttClient> _instance;
  static std::once_flag _instantiated;

  static std::atomic_bool _connected;

  explicit MqttClient(MqttClientConfiguration &config);

  MqttClientConfiguration &_config;
  std::future<void> _connection_future;
  std::unique_ptr<mqtt::async_client> _client;

  bool check_connected();
};

} // namespace pc
