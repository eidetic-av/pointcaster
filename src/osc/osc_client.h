#pragma once

#include "../logger.h"
#include "../publisher/publishable_traits.h"
#include "../publisher/publisher_utils.h"
#include "../string_utils.h"
#include "../structs.h"
#include "osc_client_config.gen.h"
#include <array>
#include <atomic>
#include <concurrentqueue/blockingconcurrentqueue.h>
#include <fmt/format.h>
#include <future>
#include <lo/lo_cpp.h>
#include <map>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>


namespace pc::osc {

class OscClient {
public:
  OscClient(OscClientConfiguration &config);
  ~OscClient();

  OscClient(const OscClient &) = delete;
  OscClient &operator=(const OscClient &) = delete;
  OscClient(OscClient &&) = delete;
  OscClient &operator=(OscClient &&) = delete;

  template <typename T>
  void publish(const std::string_view topic, const T &data,
               std::initializer_list<std::string_view> topic_nodes = {}) {

    if (!_config.enable) return;

    // if we're trying to publish a container
    if constexpr (is_publishable_container_v<T>) {
      // if the container contains scalar types
      if constexpr (pc::types::ScalarType<typename T::value_type>) {
        // add all elements to the message
        lo::Message msg;
        for (auto &element : data) { msg.add(element); }
        _messages_to_publish.enqueue(
            {.address = publisher::construct_topic_string(topic, topic_nodes),
             .value = msg});
      }
      // if the container contains other containers, recurse into this function
      else if constexpr (is_publishable_container<typename T::value_type>()) {
        std::size_t i = 0;
        for (auto &element : data) {
          auto element_topic =
              pc::strings::concat(topic, fmt::format("/{}", i++));
          publish(element_topic, element, topic_nodes);
        }
      }
    }
    // if we're trying to publish a simple scalar type
    else if constexpr (pc::types::ScalarType<T>) {
      lo::Message msg;
      msg.add(data);
      _messages_to_publish.enqueue(
          {.address = publisher::construct_topic_string(topic, topic_nodes),
           .value = msg});
    } else {
      pc::logger->error(
          "Attempting to publish an unimplemented type through OSC ({})",
          typeid(T).name());
    }
  }

  void draw_imgui_window();

private:
  OscClientConfiguration &_config;

  struct OscMessage {
    std::string address;
    lo::Message value;
  };

  std::jthread _sender_thread;
  moodycamel::BlockingConcurrentQueue<OscMessage> _messages_to_publish{};

  void send_messages(std::stop_token st);
};

} // namespace pc::osc
