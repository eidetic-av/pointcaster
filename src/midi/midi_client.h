#pragma once

#include "../logger.h"
#include "../publisher/publishable_traits.h"
#include "../publisher/publisher_utils.h"
#include "../string_utils.h"
#include "../structs.h"
#include "midi_client_config.gen.h"
#include <array>
#include <atomic>
#include <concurrentqueue/blockingconcurrentqueue.h>
#include <expected>
#include <format>
#include <future>
#include <libremidi/client.hpp>
#include <libremidi/libremidi.hpp>
#include <map>
#include <memory>
#include <msgpack.hpp>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>

namespace pc::midi {

static inline MidiOutputRoute last_output_route{1, 1};

class MidiClient {
public:
  MidiClient(MidiClientConfiguration &config);
  ~MidiClient();

  MidiClient(const MidiClient &) = delete;
  MidiClient &operator=(const MidiClient &) = delete;
  MidiClient(MidiClient &&) = delete;
  MidiClient &operator=(MidiClient &&) = delete;

  void publish(const std::string_view topic, const uint8_t channel_num,
	       const uint8_t cc_num, const float value);

  template <typename T>
  std::enable_if_t<is_publishable_container_v<T>, void>
  publish(const std::string_view topic, const T &data,
          std::initializer_list<std::string_view> topic_nodes = {}) {

    std::size_t i = 0;
    for (auto &element : data) {
      auto element_topic = pc::strings::concat(topic, std::format("/{}", i++));
      if constexpr (is_publishable_container<typename T::value_type>()) {
	publish(element_topic, element, topic_nodes);
      } else {
	auto flattened_topic =
	    publisher::construct_topic_string(element_topic, topic_nodes);

	auto [route_itr, created_new_route] =
	    _config.output_routes.try_emplace(flattened_topic);

        auto &route = route_itr->second;

        if (created_new_route) {
          // New route created for this topic, set it to use last_output_route
          // and increment that
          route = last_output_route;
          if (last_output_route.cc == 129) {
            last_output_route.cc = 1;
            if (++last_output_route.channel == 17)
              last_output_route.channel = 1;
          } else {
            last_output_route.cc++;
          }
        }
	route.last_value = element;

	publish(flattened_topic, route.channel, route.cc, route.last_value);
      }
    }
  }

  void draw_imgui_window();

private:
  MidiClientConfiguration &_config;

  std::unordered_map<std::string, libremidi::observer> _midi_observers;
  std::unordered_map<std::string, libremidi::midi_in> _midi_inputs;

  libremidi::midi_out _virtual_output;

  struct MidiOutMessage {
    std::string topic;
    uint8_t channel_num;
    uint8_t cc_num;
    uint8_t value;
  };

  std::jthread _sender_thread;
  moodycamel::BlockingConcurrentQueue<MidiOutMessage>
      _messages_to_publish;

  void send_messages(std::stop_token st);

  void handle_input_added(const libremidi::input_port &port,
                          const libremidi::API &api);
  void handle_input_removed(const libremidi::input_port &port);

  void handle_input_message(const std::string &port_name,
			    const libremidi::message &msg);

  void handle_output_added(const libremidi::output_port &port,
			   const libremidi::API &api);
  void handle_output_removed(const libremidi::output_port &port);
};

} // namespace pc::midi
