#pragma once
#include <rfl/DefaultVal.hpp>

namespace pc::publishers {

struct MqttClientConfiguration {
  rfl::DefaultVal<bool> enabled = false;
  rfl::DefaultVal<std::string> broker_uri = "tcp://localhost:1884";
  rfl::DefaultVal<bool> publish_empty_stream = false;
  rfl::DefaultVal<bool> publish_empty_once = false;
};

} // namespace pc::publishers
