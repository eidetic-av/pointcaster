#pragma once

#include "../client_sync/sync_server.h"
#include "../midi/midi_device.h"
#include "../mqtt/mqtt_client.h"

#include "../operators/operator_friendly_names.h"

#ifdef WITH_OSC
#include "../osc/osc_client.h"
#endif

#include "publishable_traits.h"
#include <initializer_list>
#include <type_traits>
#include <variant>

namespace pc::publisher {

// A variant that holds all of our publisher types
using Publisher = std::variant<client_sync::SyncServer *
#ifdef WITH_MQTT
                               ,
                               mqtt::MqttClient *
#endif
#ifdef WITH_MIDI
                               ,
                               midi::MidiDevice *
#endif
#ifdef WITH_OSC
                               ,
                               osc::OscClient *
#endif
                               >;

// The list of all publishers in the application
extern std::vector<Publisher> _instances;
void add(Publisher publisher);
void remove(Publisher publisher);

// Publish a single value on a specific topic through all publishers
template <typename T>
void publish_all(const std::string_view topic, const T &data,
                 std::initializer_list<std::string_view> topic_nodes = {}) {

  // TODO this lookup and swap of id to friendly name is probably slow
  // must refactor later... after benching...

  std::string publish_label;
  auto first_node_end = topic.find('/');
  auto first_node = topic.substr(0, first_node_end);
  auto friendly_name = pc::operators::get_operator_friendly_name(first_node);
  if (!friendly_name.empty()) {
    publish_label = friendly_name + std::string(topic.substr(first_node_end));
  } else {
    publish_label = topic;
  }

#ifndef NDEBUG
  pc::logger->debug("Publishing with label: {}", publish_label);
  if (topic_nodes.size() > 0) {
    pc::logger->debug("- Topic nodes: ", publish_label);
    std::for_each(topic_nodes.begin(), topic_nodes.end(),
                  [](const auto &node) { pc::logger->debug("  - {}", node); });
  }
#endif

  for (auto &publisher : _instances) {
    std::visit(
        [&publish_label, &data, &topic_nodes](auto &publisher) {
          publisher->publish(publish_label, data, topic_nodes);
        },
        publisher);
  }
}

} // namespace pc::publisher
