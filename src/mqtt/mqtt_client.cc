#include "mqtt_client.h"
#include <imgui.h>
#include <imgui_stdlib.h>
#include <regex>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <chrono>
#include <future>
#include <stdexcept>

namespace pc {

using namespace std::chrono_literals;

static const std::regex uri_pattern(R"(^([a-zA-Z]+):\/\/([^:\/]+):(\d+)$)");

std::shared_ptr<MqttClient> MqttClient::_instance;
std::mutex MqttClient::_mutex;

std::string input_hostname;
int input_port;

MqttClient::MqttClient(MqttClientConfiguration &config) : _config(config) {
  set_uri(config.broker_uri);
  if (_config.auto_connect) connect();
}

MqttClient::~MqttClient() {
  spdlog::info("Disconnecting");
  _client->disconnect()->wait();
  spdlog::info("Disconnected");
}

void MqttClient::connect() {
  connect(_config.broker_uri);
}

void MqttClient::connect(const std::string_view uri) {
  const std::string uri_str {uri};
  std::smatch match;
  if (!std::regex_search(uri_str, match, uri_pattern)) {
    const auto error_msg = fmt::format("Invalid MQTT URI: {}", uri);
    spdlog::error(error_msg);
    throw std::invalid_argument(error_msg);
  }
  try {
    const auto host_name = match.str(2);
    const auto port = std::stoi(match.str(3));
    connect(host_name, port);
  } catch (std::exception &e) {
    spdlog::error(e.what());
    throw;
  }
}

void MqttClient::connect(const std::string_view host_name, const int port) {
  const auto previous_uri = _config.broker_uri;
  set_uri(host_name, port);

  if (_config.broker_uri != previous_uri) {
    disconnect();
  }

  spdlog::info("Connecting to MQTT broker at {}", _config.broker_uri);

  _client = std::make_unique<mqtt::async_client>(_config.broker_uri.data(),
						 _config.client_id.data());

  auto options = mqtt::connect_options_builder()
		     .clean_session()
		     .automatic_reconnect()
		     .finalize();

  _connection_future = std::async(std::launch::async, [this, options]() {
    if (_client->connect(options)->wait_for(10s)) {
      spdlog::info("MQTT connection active");
    } else
      spdlog::warn("MQTT connection failed");
  });
}

void MqttClient::disconnect() {
  if (!_client) return;
  try {
    _client->disconnect()->wait();
  } catch (mqtt::exception &e) {
    spdlog::error(e.what());
  }
  _client.reset();
  spdlog::info("Disconnected from MQTT broker");
}

bool MqttClient::check_connected() {
  if (!_client) {
    spdlog::warn("Attempt to publish with disconnected MQTT client");
    return false;
  }
  return true;
}

void MqttClient::publish(const std::string_view topic,
                         const std::string_view message) {
  if (!check_connected()) return;
  mqtt::topic t(*_client.get(), topic.data());
  t.publish(message.data());
}

void MqttClient::publish(const std::string_view topic,
			 const std::vector<uint8_t>& payload) {
  if (!check_connected()) return;
  mqtt::topic t(*_client.get(), topic.data());
  t.publish(payload.data(), payload.size());
}

void MqttClient::set_uri(const std::string_view uri) {
  const std::string uri_str { uri };
  std::smatch match;
  if (!std::regex_search(uri_str, match, uri_pattern)) {
    spdlog::error("Invalid MQTT URI: {}", uri);
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
