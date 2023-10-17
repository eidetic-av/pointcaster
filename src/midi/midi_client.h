#pragma once

#include "../logger.h"
#include "../publisher/publishable_traits.h"
#include "../publisher/publisher_utils.h"
#include "../structs.h"
#include "midi_client_config.h"
#include <array>
#include <atomic>
#include <concurrentqueue/blockingconcurrentqueue.h>
#include <expected>
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

  // TODO generic length array

  void publish(const std::string_view topic, const uint8_t channel_num,
               const std::array<float, 2> values);

  void publish(const std::string_view topic, const uint8_t channel_num,
               const std::array<float, 3> values);

  void publish(const std::string_view topic, const uint8_t channel_num,
               const std::array<float, 4> values);

  template <typename T>
  std::enable_if_t<is_publishable_container_v<T>, void>
  publish(const std::string_view topic, const T &data,
          std::initializer_list<std::string_view> topic_nodes = {}) {

    if constexpr (std::is_same_v<typename T::value_type,
				 std::vector<std::array<float, 2>>>) {
      pc::logger->warn(
	  "MIDI publishing not yet implemented for nested containers");
      return;
    } else {

      auto flattened_topic =
          publisher::construct_topic_string(topic, topic_nodes);
      // for publishable containers, send out each element on a unique channel
      // for testing
      auto channel_num = 1;
      for (auto &element : data) {
        publish(flattened_topic, channel_num, element);
        if (++channel_num > 16)
          break;
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
  moodycamel::BlockingConcurrentQueue<const MidiOutMessage>
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
