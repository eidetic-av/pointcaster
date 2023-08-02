#pragma once

#include <string>

struct MqttClientConfiguration {
  std::string client_id = "pointcaster";
  std::string broker_uri = "tcp://localhost:1884";
  bool auto_connect = false;
};
