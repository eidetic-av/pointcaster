#pragma once

#include <string>

namespace pc::mqtt {

struct MqttClientConfiguration {
  bool show_window = false;
  std::string id = "pointcaster";
  std::string broker_uri = "tcp://localhost:1884";
  bool auto_connect = false;
  bool publish_empty_stream = false;
  bool publish_empty_once = false;
};

} // namespace pc::mqtt
