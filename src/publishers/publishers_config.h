#pragma once
#include "message_streamer/message_streamer_config.h"
#include "mqtt/mqtt_client_config.h"
#include "osc/osc_sender_config.h"

#include <rfl/DefaultVal.hpp>

namespace pc::publishers {

struct PublishersConfiguration {
  rfl::DefaultVal<int> publish_hz = 100; // @minmax(5, 200)

  rfl::DefaultVal<MessageStreamerConfiguration> message_streamer;
  rfl::DefaultVal<MqttClientConfiguration> mqtt;
  rfl::DefaultVal<OscSenderConfiguration> osc;
};

} // namespace pc::publishers
