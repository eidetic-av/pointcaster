#include "mqtt_client.h"
#include <imgui.h>
#include <imgui_stdlib.h>
#include <regex>
#include <fmt/format.h>
#include <chrono>
#include <future>
#include <stdexcept>
#include <msgpack.hpp>
#include "../logger.h"

namespace pc {

using namespace std::chrono_literals;

static const std::regex uri_pattern(R"(^([a-zA-Z]+):\/\/([^:\/]+):(\d+)$)");

std::shared_ptr<MqttClient> MqttClient::_instance;
std::mutex MqttClient::_mutex;
std::atomic_bool MqttClient::_connected = false;

std::string input_hostname;
int input_port;

MqttClient::MqttClient(MqttClientConfiguration &config) : _config(config) {
  set_uri(config.broker_uri);
  if (_config.auto_connect) connect();
}

MqttClient::~MqttClient() {
  if (!_client) return;
  _client->disconnect()->wait();
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

  _client = std::make_unique<mqtt::async_client>(_config.broker_uri.data(),
						 _config.client_id.data());

  auto options = mqtt::connect_options_builder()
		     .clean_session()
		     .automatic_reconnect()
		     .finalize();

  _connection_future = std::async(std::launch::async, [this, options]() {
    if (_client->connect(options)->wait_for(10s)) {
      MqttClient::_connected = true;
      pc::logger->info("MQTT connection active");
    } else {
      pc::logger->warn("MQTT connection failed");
      MqttClient::_connected = false;
    }
  });
}

void MqttClient::disconnect() {
  if (!_client) return;
  try {
    _client->disconnect()->wait();
    _client.reset();
    MqttClient::_connected = false;
    pc::logger->info("Disconnected from MQTT broker");
  } catch (mqtt::exception &e) {
    pc::logger->error(e.what());
  }
}

void MqttClient::publish(const std::string_view topic,
                         const std::string_view message) {
  if (!_connected) return;
  mqtt::topic t(*_client.get(), topic.data());
  t.publish(message.data());
}

void MqttClient::publish(const std::string_view topic,
			 const std::vector<uint8_t> &payload) {
  publish(topic, payload.data(), payload.size());
}

void MqttClient::publish(const std::string_view topic, const void *payload,
                         const std::size_t size) {
  if (!_connected) return;
  mqtt::topic t(*_client.get(), topic.data());
  t.publish(payload, size);
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

void MqttClient::draw_imgui_window() {
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("MQTT", nullptr);
  ImGui::InputText("Broker hostname", &input_hostname);
  ImGui::InputInt("Broker port", &input_port);
  ImGui::Spacing();

  const auto draw_connect_buttons = [this]() {
    if (ImGui::Button("Connect"))
      connect(input_hostname, input_port);
    ImGui::SameLine();
    if (ImGui::Button("Disconnect"))
      disconnect();
  };

  if (!_connection_future.valid()) {
    draw_connect_buttons();
  } else if (_connection_future.wait_for(0s) == std::future_status::ready) {
    draw_connect_buttons();
  } else {
    ImGui::BeginDisabled();
    draw_connect_buttons();
    ImGui::EndDisabled();
  }

  ImGui::Checkbox("Auto connect", &_config.auto_connect);
  ImGui::End();
}

} // namespace pc
