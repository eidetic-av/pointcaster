#include "mqtt_client.h"
#include "../logger.h"
#include "../publisher.h"
#include <chrono>
#include <fmt/format.h>
#include <future>
#include <imgui.h>
#include <imgui_stdlib.h>
#include <msgpack.hpp>
#include <regex>
#include <stdexcept>

namespace pc {

using namespace std::chrono_literals;

static const std::regex uri_pattern(R"(^([a-zA-Z]+):\/\/([^:\/]+):(\d+)$)");

std::shared_ptr<MqttClient> MqttClient::_instance;
std::once_flag MqttClient::_instantiated;

std::string input_hostname;
int input_port;

void MqttClient::create(MqttClientConfiguration &config) {
  std::call_once(_instantiated, [&config]() {
    _instance = std::shared_ptr<MqttClient>(new MqttClient(config));
    publisher::add(_instance);
  });
}

MqttClient::MqttClient(MqttClientConfiguration &config) : _config(config) {
  set_uri(config.broker_uri);
  if (_config.auto_connect) connect();
  _queue_producer_token =
      std::make_unique<moodycamel::ProducerToken>(_messages_to_publish);
  _sender_thread = std::jthread([this](auto st) { send_messages(st); });
}

MqttClient::~MqttClient() {
  if (!_client) return;
  _sender_thread.request_stop();
  _sender_thread.join();
  _client->disconnect();
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

  _client = std::make_unique<mqtt::client>(_config.broker_uri.data(),
						 _config.id.data());

  auto options = mqtt::connect_options_builder()
		     .connect_timeout(5ms)
		     .clean_session()
		     .automatic_reconnect()
		     .finalize();

  _connection_future = std::async(std::launch::async, [this, options]() {
    _connecting = true;
    try {
      _client->connect(options);
      pc::logger->info("MQTT connection active");
    } catch (const mqtt::exception &e) {
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
    } catch (mqtt::exception &e) {
      pc::logger->error(e.what());
    }
  });
}

void MqttClient::publish(const std::string_view topic,
                         const std::string_view message) {
  publish(topic, message.data(), message.size());
}

void MqttClient::publish(const std::string_view topic,
			 const std::vector<uint8_t> &payload) {
  publish(topic, payload.data(), payload.size());
}

void MqttClient::publish(const std::string_view topic, const void *payload,
                         const std::size_t size) {
  auto data = static_cast<const std::byte *>(payload);
  std::vector<std::byte> buffer(data, data + size);
  _messages_to_publish.enqueue(*_queue_producer_token,
			       {std::string(topic), std::move(buffer)});
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

  while (!st.stop_requested()) {

    auto message_count = _messages_to_publish.size_approx();
    if (message_count == 0) continue;

    std::vector<MqttMessage> messages;
    messages.reserve(message_count);

    _messages_to_publish.try_dequeue_bulk_from_producer(
	*_queue_producer_token, std::back_inserter(messages), message_count);

    if (!_client || !_client->is_connected()) {
      continue;
    }

    for (auto &message : messages) {
      auto &payload = message.buffer;
      try {
	_client->publish(message.topic, payload.data(), payload.size());
      } catch (const mqtt::exception &e) {
	pc::logger->error(e.what());
      }
    }
  }
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

  ImGui::Checkbox("Publish empty data", &_config.publish_empty_data);

  ImGui::End();
}

} // namespace pc
