#pragma once

#include "../logger.h"
#include "../publisher/publishable_traits.h"
#include "../publisher/publisher_utils.h"
#include "../string_utils.h"
#include "../structs.h"
#include "midi_device_config.gen.h"
#include "rtpmidi_device.h"
#include <array>
#include <atomic>
#include <concurrentqueue/blockingconcurrentqueue.h>
#include <expected>
#include <fmt/format.h>
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

class MidiDevice {
public:
  MidiDevice(MidiDeviceConfiguration &config);
  ~MidiDevice();

  MidiDevice(const MidiDevice &) = delete;
  MidiDevice &operator=(const MidiDevice &) = delete;
  MidiDevice(MidiDevice &&) = delete;
  MidiDevice &operator=(MidiDevice &&) = delete;

  void publish(libremidi::message_type, const uint8_t channel_num,
	       const uint8_t cc_or_note_num, const uint8_t value);

  void publish(const std::string_view topic, const MidiOutputRoute& route);

  template <typename T>
  std::enable_if_t<pc::types::ScalarType<T>, void>
  publish(const std::string_view topic, const T &data,
          std::initializer_list<std::string_view> topic_nodes = {}) {
    auto flattened_topic =
        publisher::construct_topic_string(topic, topic_nodes);
    auto &route = get_or_create_output_route(flattened_topic);
    route.last_value = data;
    publish(flattened_topic, route);
  }

  template <typename T>
  std::enable_if_t<is_publishable_container_v<T>, void>
  publish(const std::string_view topic, const T &data,
	  std::initializer_list<std::string_view> topic_nodes = {}) {
    std::size_t i = 0;
    for (auto &element : data) {
      auto element_topic = pc::strings::concat(topic, fmt::format("/{}", i++));
      if constexpr (is_publishable_container<typename T::value_type>()) {
        publish(element_topic, element, topic_nodes);
      } else {
        auto flattened_topic =
            publisher::construct_topic_string(element_topic, topic_nodes);
        auto &route = get_or_create_output_route(flattened_topic);
        route.last_value = element;
        publish(flattened_topic, route);
      }
    }
  }

  void draw_imgui_window();

private:
  MidiDeviceConfiguration &_config;
  std::optional<std::reference_wrapper<MidiOutputRoute>> _selected_output_route;

  std::unordered_map<std::string, libremidi::observer> _midi_observers;
  std::unordered_map<std::string, libremidi::midi_in> _midi_inputs;

  libremidi::midi_out _virtual_output;

  std::unique_ptr<RtpMidiDevice> _rtp_midi;

  std::jthread _sender_thread;

  struct MidiOutMessage {
    libremidi::message_type type;
    uint8_t channel_num;
    uint8_t cc_or_note_num;
    uint8_t value;
  };
  moodycamel::BlockingConcurrentQueue<MidiOutMessage>
      _messages_to_publish;

  void send_messages(std::stop_token st);


  MidiOutputRoute &get_or_create_output_route(std::string_view topic) {
    static MidiOutputRoute last_output_route{1, 1};

    auto [route_itr, created_new_route] = _config.output_routes.try_emplace(
        std::string{topic.data(), topic.size()});
    auto &route = route_itr->second;

    if (created_new_route) {
      route = last_output_route;
      if (last_output_route.cc == 129) {
        last_output_route.cc = 1;
        if (++last_output_route.channel == 17)
          last_output_route.channel = 1;
      } else {
        last_output_route.cc++;
      }
    }
    return route;
  }

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
