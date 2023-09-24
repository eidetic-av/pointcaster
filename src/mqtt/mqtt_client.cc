#include "mqtt_client.h"
#include "../logger.h"
#include "../publisher.h"
#include <chrono>
#include <fmt/format.h>
#include <future>
#include <imgui.h>
#include <imgui_stdlib.h>
#include <initializer_list>
#include <msgpack.hpp>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <unordered_map>

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
    std::atexit([] {
      publisher::remove(_instance);
      _instance.reset();
    });
  });
}

MqttClient::MqttClient(MqttClientConfiguration &config) : _config(config) {
  set_uri(config.broker_uri);
  if (_config.auto_connect) connect();
  _sender_thread = std::jthread([this](auto st) { send_messages(st); });
}

MqttClient::~MqttClient() {
  if (!_client) return;
  _sender_thread.request_stop();
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
			 const std::string_view message,
			 std::initializer_list<std::string_view> topic_nodes) {
  publish(topic, message.data(), message.size(), topic_nodes);
}

void MqttClient::publish(const std::string_view topic,
                         const std::vector<uint8_t> &payload,
                         std::initializer_list<std::string_view> topic_nodes) {
  publish(topic, payload.data(), payload.size(), topic_nodes);
}

void MqttClient::publish(const std::string_view topic, const void *payload,
                         const std::size_t size, std::initializer_list<std::string_view> topic_nodes) {
  auto data = static_cast<const std::byte *>(payload);
  std::vector<std::byte> buffer(data, data + size);
  auto topic_string = construct_topic_string(topic, topic_nodes);
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

  while (!st.stop_requested()) {

    auto message_count = _messages_to_publish.size_approx();
    if (message_count == 0) continue;

    std::vector<MqttMessage> messages;
    messages.reserve(message_count);

    _messages_to_publish.try_dequeue_bulk(std::back_inserter(messages),
					  message_count);

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

  pc::logger->info("Ended MQTT sender thread");
}

std::string MqttClient::construct_topic_string(
    const std::string_view topic_name,
    std::initializer_list<std::string_view> topic_nodes) {

  // a map to cache pre-constructed topic strings avoiding reconstruction
  static std::unordered_map<std::size_t, std::string> cache;
  static std::mutex cache_mutex;

  // compute the hash of the arguments
  std::hash<std::string_view> hasher;
  std::size_t hash = hasher(topic_name);
  for (const auto &node : topic_nodes) {
    // mix hash bits with each topic node
    hash ^= hasher(node) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }

  // check and return if our result is already cached
  {
    std::lock_guard lock(cache_mutex);
    auto it = cache.find(hash);
    if (it != cache.end()) {
      return it->second;
    }
  }

  // if it's not cached we construct the string

  // allocate the memory
  std::size_t result_string_length = topic_name.size();
  for (const auto &node : topic_nodes) {
    result_string_length += node.size() + 1; // (+1 for the delimiter character)
  }

  // concatenate the string with a '/' as delimiter
  std::string result;
  result.reserve(result_string_length);
  for (const auto &node : topic_nodes) {
    result.append(node);
    result.push_back('/');
  }
  result.append(topic_name);

  // add the new string to our cache
  {
    std::lock_guard lock(cache_mutex);
    cache[hash] = result;
  }

  return result;
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
