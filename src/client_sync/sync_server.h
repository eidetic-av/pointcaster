#pragma once

#include "../logger.h"
#include "../publisher/publishable_traits.h"
#include "../publisher/publisher_utils.h"
#include "../string_utils.h"
#include "../structs.h"
#include "sync_server_config.gen.h"
#include <thread>
#include <type_traits>
#include <variant>
#include <zmq.hpp>
#ifndef __CUDACC__
#include <concurrentqueue/blockingconcurrentqueue.h>
#include <zpp_bits.h>
#endif
#include <unordered_set>

// TODO: put these in a util class
// Helper template to extract types from a variant
template <typename... Ts> struct extract_variant_types {};

template <typename T, typename... Ts>
struct extract_variant_types<std::variant<T, Ts...>> {
  using type = std::tuple<T, Ts...>;
};

// Helper function to check if a type exists within a tuple of types
template <typename T, typename Tuple> constexpr bool is_in_tuple = false;

template <typename T, typename Head, typename... Tail>
constexpr bool is_in_tuple<T, std::tuple<Head, Tail...>> =
    std::is_same_v<T, Head> || is_in_tuple<T, std::tuple<Tail...>>;

/////////////////

namespace pc::client_sync {

using array3f = std::array<float, 3>;
using pc::types::Float3;

struct ParameterUpdate {
  std::string id;
  std::variant<float, int, Float3, array3f> value;
};

enum class MessageType : uint8_t {
  Connected = 0x00,
  ClientHeartbeat = 0x01,
  ClientHeartbeatResponse = 0x02,
  ParameterUpdate = 0x10,
  ParameterRequest = 0x11
};

using ServerToClientMessage = std::variant<MessageType, ParameterUpdate>;

class SyncServer {
public:
  SyncServer(SyncServerConfiguration &config);
  ~SyncServer();

  SyncServer(const SyncServer &) = delete;
  SyncServer &operator=(const SyncServer &) = delete;

  SyncServer(SyncServer &&) = delete;
  SyncServer &operator=(SyncServer &&) = delete;

  template <typename T>
  void publish(const std::string_view topic, const T &data,
	       std::initializer_list<std::string_view> topic_nodes = {}) {
    if constexpr (std::disjunction_v<std::is_same<T, float>,
				     std::is_same<T, Float3>,
				     std::is_same<T, array3f>>) {
      const auto id = publisher::construct_topic_string(topic, topic_nodes);
      enqueue_parameter_update(ParameterUpdate{id, {data}});
    }
  }

  void draw_imgui_window();

private:
  SyncServerConfiguration &_config;

  std::unique_ptr<zmq::context_t> _socket_context;
  std::unique_ptr<zmq::socket_t> _socket;

  std::unique_ptr<std::jthread> _publish_thread;
  std::unique_ptr<std::jthread> _listener_thread;

  std::unordered_set<std::string> _connected_client_ids;

#ifndef __CUDACC__
  moodycamel::BlockingConcurrentQueue<ServerToClientMessage>
      _messages_to_send;
#endif

  void enqueue_parameter_update(ParameterUpdate update);
  void enqueue_signal(MessageType signal);
};

} // namespace pc::client_sync
