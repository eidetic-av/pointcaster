#pragma once

#include "mqtt_client_config.h"
#include <atomic>
#include <expected>
#include <future>
#include <memory>
#include <mqtt/client.h>
#include <msgpack.hpp>
#include <mutex>
#include <string_view>
#include <concurrentqueue/concurrentqueue.h>
#include <span>

namespace pc {

class MqttClient {
public:

  static std::shared_ptr<MqttClient> &instance() {
    if (!_instance)
      throw std::logic_error(
	  "No MqttClient instance created. Call MqttClient::create() first.");
    return _instance;
  }

  static void create(MqttClientConfiguration &config);

  MqttClient(const MqttClient &) = delete;
  MqttClient &operator=(const MqttClient &) = delete;

  ~MqttClient();

  void connect();
  void connect(const std::string_view uri);
  void connect(const std::string_view host_name, const int port);

  void disconnect();

  bool connecting() { return _connecting; };

  void set_uri(const std::string_view uri);
  void set_uri(const std::string_view hostname, int port);

  void publish(const std::string_view topic, const std::string_view message,
	       std::initializer_list<std::string_view> topic_nodes = {});
  void publish(const std::string_view topic, const std::vector<uint8_t> &bytes,
	       std::initializer_list<std::string_view> topic_nodes = {});
  void publish(const std::string_view topic, const void *payload,
	       const std::size_t size,
	       std::initializer_list<std::string_view> topic_nodes = {});

  template <typename T>
  void publish(const std::string_view topic, const T &data,
	       std::initializer_list<std::string_view> topic_nodes = {}) {
    msgpack::sbuffer send_buffer;
    msgpack::pack(send_buffer, data);
    publish(topic, send_buffer.data(), send_buffer.size(), topic_nodes);
  }

  bool publish_empty() { return _config.publish_empty_data; }

  void draw_imgui_window();

private:
  static std::shared_ptr<MqttClient> _instance;
  static std::once_flag _instantiated;

  explicit MqttClient(MqttClientConfiguration &config);

  MqttClientConfiguration &_config;
  std::future<void> _connection_future;
  std::atomic_bool _connecting;
  std::unique_ptr<mqtt::client> _client;

  struct MqttMessage {
    std::string topic;
    std::vector<std::byte> buffer;
  };

  std::jthread _sender_thread;
  moodycamel::ConcurrentQueue<const MqttMessage> _messages_to_publish;

  void send_messages(std::stop_token st);

  std::string
  construct_topic_string(const std::string_view topic_name,
			 std::initializer_list<std::string_view> topic_nodes);
};

} // namespace pc
