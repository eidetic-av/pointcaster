#pragma once
#include <rfl/DefaultVal.hpp>
#include <string>

namespace pc::publishers {

struct MqttClientConfiguration {
  rfl::DefaultVal<bool> enabled = false;
  rfl::DefaultVal<std::string> broker_uri = "tcp://localhost:1884";
  rfl::DefaultVal<std::string> client_id = "pointcaster";
  rfl::DefaultVal<bool> auto_reconnect = true;

  // how a value that isn't a plain scalar or string gets encoded
  enum class SerializationFormat { JSON, MessagePack };
  rfl::DefaultVal<SerializationFormat> serialization_format =
      SerializationFormat::JSON;

  rfl::DefaultVal<bool> serialize_as_structures = false;
  rfl::DefaultVal<bool> send_retained = false;

  enum class EmptyMessageHandling {
    Ignore,
    PublishEmptyOnce,
    PublishEmptyAlways
  };
  rfl::DefaultVal<EmptyMessageHandling> empty_message_handling =
      EmptyMessageHandling::Ignore;

};

} // namespace pc::publishers
