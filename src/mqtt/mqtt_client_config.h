#pragma once

#include <string>

#include <serdepp/serde.hpp>

struct MqttClientConfiguration {
  bool show_window = false;
  std::string id = "pointcaster";
  std::string broker_uri = "tcp://localhost:1884";
  bool auto_connect = false;
  bool publish_empty_data = false;

  DERIVE_SERDE(MqttClientConfiguration,
	       (&Self::show_window, "show_window")
	       (&Self::id, "id") (&Self::broker_uri, "broker_uri")
	       (&Self::auto_connect, "auto_connect")
	       (&Self::publish_empty_data, "publish_empty_data"))
};
