#include "mqtt_client.h"
#include "../logger.h"
#include "../publisher/publisher.h"
#include "../publisher/publisher_utils.h"
#include <chrono>
#include <fmt/format.h>
#include <functional>
#include <future>
#include <imgui.h>
#include <imgui_stdlib.h>
#include <initializer_list>
#include <msgpack.hpp>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <unordered_map>

namespace pc::mqtt {

using namespace std::chrono_literals;

static const std::regex uri_pattern(R"(^([a-zA-Z]+):\/\/([^:\/]+):(\d+)$)");

std::string input_hostname;
int input_port;

MqttClient::MqttClient(MqttClientConfiguration &config)
    : _config(config), _sender_thread([this](auto st) { send_messages(st); }) {
  set_uri(config.broker_uri);
  if (_config.auto_connect) connect();
  publisher::add(this);
}

MqttClient::~MqttClient() {
  _sender_thread.request_stop();
  _sender_thread.join();
  publisher::remove(this);
  if (_client) _client->disconnect();
}

void MqttClient::connect() {
  connect(_config.broker_uri);
}

void MqttClient::connect(const std::string_view uri) {
  const std::string uri_str {uri};
  std::smatch match;
  if (!std::regex_search(uri_str, match, uri_pattern)) {
    const auto error_msg = fmt::format("Invalid MQTT URI: {}", uri);
    pc::logger->error(error_msg);
    throw std::invalid_argument(error_msg);
  }
  try {
    const auto host_name = match.str(2);
    const auto port = std::stoi(match.str(3));
    connect(host_name, port);
  } catch (std::exception &e) {
    pc::logger->error(e.what());
    throw;
  }
}

void MqttClient::connect(const std::string_view host_name, const int port) {
  const auto previous_uri = _config.broker_uri;
  set_uri(host_name, port);

  if (_config.broker_uri != previous_uri) {
    disconnect();
  }

  pc::logger->info("Connecting to MQTT broker at {}", _config.broker_uri);

  _client = std::make_unique<::mqtt::client>(_config.broker_uri.data(),
						 _config.id.data());

  auto options = ::mqtt::connect_options_builder()
		     .connect_timeout(150ms)
		     .clean_session()
		     .automatic_reconnect()
		     .finalize();

  _connection_future = std::async(std::launch::async, [this, options]() {
    _connecting = true;
    try {
      _client->connect(options);
      pc::logger->info("MQTT connection active");
    } catch (const ::mqtt::exception &e) {
      pc::logger->error(e.what());
    }
    _connecting = false;
  });
}

void MqttClient::disconnect() {
  if (!_client) return;
  _connection_future = std::async(std::launch::async, [this]() {
    try {
      _client->disconnect();
      pc::logger->info("Disconnected from MQTT broker");
      _client.reset();
    } catch (::mqtt::exception &e) {
      pc::logger->error(e.what());
    }
  });
}

void MqttClient::publish(const std::string_view topic,
			 const std::string_view message,
			 std::initializer_list<std::string_view> topic_nodes) {
  publish(topic, message.data(), message.size(), topic_nodes);
}

void MqttClient::publish(const std::string_view topic, const void *payload,
                         const std::size_t size,
                         std::initializer_list<std::string_view> topic_nodes,
			 bool empty) {
  if (empty && !_config.publish_empty_stream && !_config.publish_empty_once) {
    return;
  }

  auto topic_string = publisher::construct_topic_string(topic, topic_nodes);

  if (_config.publish_empty_once) {
    static std::unordered_map<std::string, bool> published_empty_once;
    static std::mutex published_empty_once_mutex;
    std::lock_guard lock(published_empty_once_mutex);
    if (empty && published_empty_once[topic_string]) return;
    published_empty_once[topic_string] = empty;
  }

  auto data = static_cast<const std::byte *>(payload);
  std::vector<std::byte> buffer(data, data + size);
  _messages_to_publish.enqueue({std::move(topic_string), std::move(buffer)});
}

void MqttClient::set_uri(const std::string_view uri) {
  const std::string uri_str { uri };
  std::smatch match;
  if (!std::regex_search(uri_str, match, uri_pattern)) {
    pc::logger->error("Invalid MQTT URI: {}", uri);
    return;
  }
  _config.broker_uri = uri;
  input_hostname = match.str(2);
  input_port = std::stoi(match.str(3));
}

void MqttClient::set_uri(const std::string_view hostname, int port) {
  set_uri(fmt::format("tcp://{}:{}", hostname, port));
}

void MqttClient::send_messages(std::stop_token st) {
  using namespace std::chrono_literals;

  static constexpr auto thread_wait_time = 50ms;

  while ((!_client || !_client->is_connected()) && !st.stop_requested()) {
    std::this_thread::sleep_for(thread_wait_time);
  }

  while (!st.stop_requested()) {
    MqttMessage message;
    while (_messages_to_publish.wait_dequeue_timed(message, thread_wait_time)) {

      // messages are lost if the client is not connected
      if (!_client || !_client->is_connected()) continue;

      auto &payload = message.buffer;
      try {
	_client->publish(message.topic, payload.data(), payload.size());
      } catch (const ::mqtt::exception &e) {
        pc::logger->error(e.what());
      }
    }
  }

  pc::logger->info("Ended MQTT sender thread");
}

void MqttClient::draw_imgui_window() {
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("MQTT", nullptr);
  ImGui::InputText("Broker hostname", &input_hostname);
  ImGui::InputInt("Broker port", &input_port);
  ImGui::Spacing();

  const auto draw_connect_buttons = [this](bool connected) {
    if (connected) ImGui::BeginDisabled();
    if (ImGui::Button("Connect"))
      connect(input_hostname, input_port);
    if (connected) ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Disconnect"))
      disconnect();
  };

  if (_connecting) {
    ImGui::BeginDisabled();
    draw_connect_buttons(false);
    ImGui::EndDisabled();
  } else {
    draw_connect_buttons(_client && _client->is_connected());
  }

  ImGui::Checkbox("Auto connect", &_config.auto_connect);
  ImGui::Spacing();

  // The following options are mutually exclusive
  if (ImGui::Checkbox("Publish empty stream", &_config.publish_empty_stream)) {
    _config.publish_empty_once = false;
  }
  if (ImGui::Checkbox("Publish empty once", &_config.publish_empty_once)) {
    _config.publish_empty_stream = false;
  }

  ImGui::End();
}

} // namespace pc
