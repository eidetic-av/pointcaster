#pragma once
#include "mqtt/mqtt_client_config.h"

#include <rfl/DefaultVal.hpp>

namespace pc::publishers {

struct PublishersConfiguration {
  rfl::DefaultVal<int> publish_hz = 100; // @minmax(5, 200)

  rfl::DefaultVal<MqttClientConfiguration> mqtt;
};

} // namespace pc::publishers
