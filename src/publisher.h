#pragma once

#include <array>
#include <initializer_list>
#include <list>
#include <type_traits>
#include <variant>
#include <vector>
#include "mqtt/mqtt_client.h"

namespace pc::publisher {

// A variant that holds all of our publisher types
using Publisher = std::variant<mqtt::MqttClient *>;

// The list of all publishers in the application
extern std::vector<Publisher> _instances;
void add(Publisher publisher);
void remove(Publisher publisher);

// Checks to determine whether the data being published is a publishable
// container type

template <typename> struct is_publishable_container : std::false_type {};

template <typename T, typename Alloc>
struct is_publishable_container<std::vector<T, Alloc>> : std::true_type {};

template <typename T, typename Alloc>
struct is_publishable_container<std::list<T, Alloc>> : std::true_type {};

template <typename T, std::size_t N>
struct is_publishable_container<std::array<T, N>> : std::true_type {};

template <typename T>
inline constexpr bool is_publishable_container_v =
    is_publishable_container<T>::value;

// Publish a container of data on a specific topic through all publishers
template <typename T>
std::enable_if_t<is_publishable_container_v<T>, void>
publish_all(const std::string_view topic, const T &data,
	    std::initializer_list<std::string_view> topic_nodes = {}) {
  for (auto &publisher : _instances) {
    std::visit(
	[&topic, &data, &topic_nodes](auto &publisher) {
	  // if the container isn't empty, call publish
	  if (!data.empty()) {
            publisher->publish(topic, data, topic_nodes);
          } else if (publisher->publish_empty()) {
            // if the container is empty, we check if the publisher is set to
	    // publish empty containers or not
	    publisher->publish(topic, data, topic_nodes);
          }
        },
        publisher);
  }
}

// Publish a data on a specific topic through all publishers
template <typename T>
std::enable_if_t<!is_publishable_container_v<T>, void>
publish_all(const std::string_view topic, const T &data) {
  for (auto &publisher : _instances) {
    std::visit(
	[&topic, &data](auto &publisher) {
	  publisher->publish(topic, data);
	},
	publisher);
  }
}

} // namespace pc::publisher
