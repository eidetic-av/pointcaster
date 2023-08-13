#pragma once

#include <string>
#include <nlohmann/json.hpp>

struct MqttClientConfiguration {
  std::string client_id = "pointcaster";
  std::string broker_uri = "tcp://localhost:1884";
  bool auto_connect = false;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MqttClientConfiguration, client_id,
                                   broker_uri, auto_connect);
