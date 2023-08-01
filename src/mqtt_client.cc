#include "mqtt_client.h"
#include <spdlog/spdlog.h>

namespace pc {

MqttClient::MqttClient(std::string_view client_id, std::string_view uri)

    : _client_id(client_id), _uri(uri) {

  _client = std::make_unique<mqtt::async_client>(uri.data(), client_id.data());

  spdlog::info("Connecting to MQTT broker at {}", _client->get_server_uri());

  auto options = mqtt::connect_options_builder().clean_session().finalize();

  _client->connect(options)->wait();

  spdlog::info("Publishing test");
  publish("vertices", "woohoo");

}

MqttClient::~MqttClient() {
  spdlog::info("Disconnecting");
  _client->disconnect()->wait();
  spdlog::info("Disconnected");
}

void MqttClient::publish(const std::string_view topic,
			 const std::string_view message) {

  mqtt::topic t(*_client.get(), topic.data());
  t.publish(message.data());
}

} // namespace pc
