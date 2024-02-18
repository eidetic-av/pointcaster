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
#include <expected>
#include <fmt/format.h>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>
#include <lo/lo.h>
#include <lo/lo_cpp.h>

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
  std::enable_if_t<is_publishable_container_v<T>, void>
  publish(const std::string_view topic, const T &data,
          std::initializer_list<std::string_view> topic_nodes = {}) {

    std::size_t i = 0;
    for (auto &element : data) {
      auto element_topic = pc::strings::concat(topic, fmt::format("/{}", i++));
      if constexpr (is_publishable_container<typename T::value_type>()) {
	publish(element_topic, element, topic_nodes);
      } else {

	// TODO Taken from MIDI

	// auto flattened_topic =
	//     publisher::construct_topic_string(element_topic, topic_nodes);

	// auto [route_itr, created_new_route] =
	//     _config.output_routes.try_emplace(flattened_topic);

        // auto &route = route_itr->second;

        // if (created_new_route) {
        //   // New route created for this topic, set it to use last_output_route
        //   // and increment that
        //   route = last_output_route;
        //   if (last_output_route.cc == 129) {
        //     last_output_route.cc = 1;
        //     if (++last_output_route.channel == 17)
        //       last_output_route.channel = 1;
        //   } else {
        //     last_output_route.cc++;
        //   }
        // }
	// route.last_value = element;

	// publish(flattened_topic, route.channel, route.cc, route.last_value);
      }
    }
  }

  void draw_imgui_window();

private:
  OscClientConfiguration &_config;


  template <typename T>
  struct OscMessage {
    std::string address;
    T value;
  };

  using OscOutMessage = std::variant<OscMessage<float>, OscMessage<int>>;

  std::jthread _sender_thread;
  moodycamel::BlockingConcurrentQueue<OscOutMessage> _messages_to_publish;

  void send_messages(std::stop_token st);
  
};

} // namespace pc::midi
