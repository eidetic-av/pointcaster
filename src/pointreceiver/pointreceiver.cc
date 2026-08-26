#include "pointreceiver.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <core/logger/logger.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <ranges>
#include <readerwriterqueue/readerwriterqueue.h>
#include <set>
#include <span>
#include <spdlog/common.h>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <util/string_map.h>
#include <variant>
#include <vector>
#include <zmq.hpp>
#include <zpp_bits.h>

namespace {

// to make sure exceptions dont cross into the api, exceptional code is
// wrapped with this template:
template <class Operation>
pointreceiver_status exception_boundary(std::string_view operation_name,
                                        Operation &&operation) noexcept {
  try {
    return operation();
  } catch (const std::bad_alloc &) {
    pc::logger()->error("{}: out of memory", operation_name);
    return POINTRECEIVER_ERROR_OUT_OF_MEMORY;
  } catch (const std::exception &error) {
    pc::logger()->error("{}: {}", operation_name, error.what());
    return POINTRECEIVER_ERROR_INTERNAL;
  } catch (...) {
    pc::logger()->error("{}: unknown exception", operation_name);
    return POINTRECEIVER_ERROR_INTERNAL;
  }
}

} // namespace

namespace pc::receiver {

using namespace std::chrono;
using namespace std::chrono_literals;

// TODO these wire types are duplicated from pointcaster's side of the
// connection. they need to move to a header shared by both when the message
// receiver is brought over to the workspace publisher
enum class MessageType : uint8_t {
  Connected = 0x00,
  ClientHeartbeat = 0x01,
  ClientHeartbeatResponse = 0x02,
  ParameterUpdate = 0x10,
  ParameterRequest = 0x11
};

using ParameterVariant =
    std::variant<int, float, float2, float3, float4, position, position_bounds>;

struct ParameterUpdate {
  std::string id;
  ParameterVariant value;
};

struct EndpointUpdate {
  std::string id;
  size_t port;
  bool active;
};

using SyncMessage = std::variant<MessageType, ParameterUpdate, EndpointUpdate>;

using SubscriptionSet = std::unordered_set<std::string>;
using SubscriptionSnapshot = std::shared_ptr<const SubscriptionSet>;

struct ChannelFrame {
  zmq::message_t message;
  std::uint64_t timestamp = 0;
  bool pending = false;
  std::shared_ptr<PointCloud> cloud;
};

// the state behind the opaque handle. it lives here rather than in
// pointreceiver_context so the whole implementation stays in one namespace
struct Context {
  std::optional<std::string> client_name;

  moodycamel::BlockingReaderWriterQueue<SyncMessage> message_queue;

  // writers publish a whole new set rather than mutating the live one, so the
  // receive thread can pick up changes with a pointer compare and no lock
  std::mutex subscription_write_mutex;
  std::atomic<SubscriptionSnapshot> subscriptions{
      std::make_shared<const SubscriptionSet>()};

  std::mutex stream_mutex;
  std::condition_variable stream_cv;
  pc::StringMap<pc::receiver::ChannelFrame> channels;
  std::uint64_t stream_timestamp = 0;
  std::set<std::string> known_stream_addresses;

  // declared last so that they are stopped and joined before any of the state
  // above is destroyed
  std::jthread message_worker;
  std::jthread pointcloud_worker;
};

namespace {

// the api is consumed through hand-written declarations in other languages,
// so the enum has to stay int-sized
static_assert(sizeof(pointreceiver_status) == sizeof(int));

static_assert(sizeof(pointreceiver_position_t) == sizeof(pc::position));
static_assert(alignof(pointreceiver_position_t) == alignof(pc::position));
static_assert(offsetof(pointreceiver_position_t, x) ==
              offsetof(pc::position, x));
static_assert(offsetof(pointreceiver_position_t, y) ==
              offsetof(pc::position, y));
static_assert(offsetof(pointreceiver_position_t, z) ==
              offsetof(pc::position, z));
static_assert(sizeof(pointreceiver_color_t) == sizeof(pc::color));
static_assert(alignof(pointreceiver_color_t) == alignof(pc::color));
static_assert(offsetof(pointreceiver_color_t, r) == offsetof(pc::color, r));
static_assert(offsetof(pointreceiver_color_t, a) == offsetof(pc::color, a));

pointreceiver_message_type convert_message_type(MessageType message_type) {
  switch (message_type) {
  case MessageType::Connected:
    return POINTRECEIVER_MSG_TYPE_CONNECTED;
  case MessageType::ClientHeartbeat:
    return POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT;
  case MessageType::ClientHeartbeatResponse:
    return POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT_RESPONSE;
  case MessageType::ParameterUpdate:
    return POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE;
  case MessageType::ParameterRequest:
    return POINTRECEIVER_MSG_TYPE_PARAMETER_REQUEST;
  default:
    return POINTRECEIVER_MSG_TYPE_UNKNOWN;
  }
}

zmq::context_t &zmq_receiver_ctx() {
  constexpr auto zmq_io_thread_count = 1;
  static zmq::context_t ctx{zmq_io_thread_count};
  return ctx;
}

zmq::message_t message_for(MessageType message_type) {
  static const auto type_buffers = [] {
    std::unordered_map<MessageType, std::vector<std::byte>> buffers;
    const auto add = [&buffers](MessageType type) {
      auto [buffer, serialize] = zpp::bits::data_out();
      if (zpp::bits::failure(serialize(SyncMessage{type}))) {
        throw std::runtime_error("Failed to serialize MessageType");
      }
      buffers.emplace(type, std::move(buffer));
    };
    add(MessageType::Connected);
    add(MessageType::ClientHeartbeat);
    add(MessageType::ParameterRequest);
    return buffers;
  }();

  const auto &buffer = type_buffers.at(message_type);
  return zmq::message_t(buffer.data(), buffer.size());
}

bool copy_to_buffer(char *destination, size_t capacity,
                    std::string_view source) {
  const auto length = source.copy(destination, capacity - 1);
  destination[length] = '\0';
  return length == source.size();
}

void message_receive_loop(std::stop_token stop_token,
                          pc::receiver::Context &ctx, zmq::socket_t socket) {
  pc::logger()->trace("Beginning message receive thread");

  // TODO replace this polling just with a regular drain like the
  // point cloud loop does

  constexpr auto poll_timeout = 100ms;
  constexpr auto heartbeat_interval = 5s;
  auto last_heartbeat_send = steady_clock::now() - heartbeat_interval;

  zmq::pollitem_t socket_poll_items[] = {{socket.handle(), 0, ZMQ_POLLIN, 0}};
  zmq::message_t incoming_msg;

  while (!stop_token.stop_requested()) {
    zmq::poll(socket_poll_items, 1, poll_timeout);

    if (socket_poll_items[0].revents & ZMQ_POLLIN) {
      while (socket.recv(incoming_msg, zmq::recv_flags::none)) {
        const std::span buffer(
            static_cast<const std::byte *>(incoming_msg.data()),
            incoming_msg.size());
        zpp::bits::in deserialize(buffer);
        SyncMessage message;
        const auto result = deserialize(message);
        if (zpp::bits::failure(result)) {
          pc::logger()->warn(
              "Failed to deserialise incoming message from Pointcaster");
        } else if (!ctx.message_queue.try_enqueue(std::move(message))) {
          pc::logger()->warn(
              "Failed to enqueue incoming message from Pointcaster");
        }
      }
    }

    const auto now = steady_clock::now();
    if (now - last_heartbeat_send >= heartbeat_interval) {
      socket.send(message_for(MessageType::ClientHeartbeat),
                  zmq::send_flags::none);
      last_heartbeat_send = now;
    }
  }

  pc::logger()->trace("Message receive thread stopping");
}

void pointcloud_receive_loop(std::stop_token stop_token,
                             pc::receiver::Context &ctx, zmq::socket_t socket) {
  pc::logger()->trace("Beginning point cloud receive thread");

  // we keep a thread-local snapshot of our subscriptions collection
  // and sync it with the socket each loop
  SubscriptionSnapshot subscriptions;
  const auto sync_subscriptions = [&] {
    // sync socket subscriptions
    const auto requested_subscriptions = ctx.subscriptions.load();
    if (requested_subscriptions != subscriptions) {
      for (const auto &address : *requested_subscriptions)
        if (!subscriptions || !subscriptions->contains(address))
          socket.set(zmq::sockopt::subscribe,
                     address.empty() ? address : address + '\0');
      if (subscriptions) {
        for (const auto &address : *subscriptions)
          if (!requested_subscriptions->contains(address))
            socket.set(zmq::sockopt::unsubscribe,
                       address.empty() ? address : address + '\0');
      }
      subscriptions = requested_subscriptions;
    }
  };

  while (!stop_token.stop_requested()) {

    sync_subscriptions();

    // block for the first frame so an idle socket doesn't spin, then drain the
    // rest without blocking & dump the latest raw bytes per channel

    auto receive_flags = zmq::recv_flags::none;
    for (zmq::message_t message; socket.recv(message, receive_flags);) {
      receive_flags = zmq::recv_flags::dontwait;
      const auto frame = message.to_string_view();
      const auto separator = frame.find('\0');
      if (separator == std::string_view::npos) continue;
      const auto address = frame.substr(0, separator);
      {
        std::lock_guard lock(ctx.stream_mutex);
        auto it = ctx.channels.find(address);
        if (it == ctx.channels.end()) {
          it = ctx.channels.emplace(std::string(address), ChannelFrame{}).first;
          ctx.known_stream_addresses.emplace(address);
        }
        it->second.message = std::move(message);
        it->second.timestamp = ++ctx.stream_timestamp;
        it->second.pending = true;
      }
      ctx.stream_cv.notify_one();
    }
  }

  pc::logger()->trace("Point cloud receive thread stopping");
}

} // namespace

} // namespace pc::receiver

// to satisfy the C-api's global scope
struct pointreceiver_context : pc::receiver::Context {};

extern "C" {

const char *pointreceiver_status_string(pointreceiver_status status) {
  switch (status) {
  case POINTRECEIVER_OK:
    return "ok";
  case POINTRECEIVER_ERROR_INVALID_ARGUMENT:
    return "invalid argument";
  case POINTRECEIVER_ERROR_ALREADY_RUNNING:
    return "already running";
  case POINTRECEIVER_ERROR_NOT_RUNNING:
    return "not running";
  case POINTRECEIVER_ERROR_CONNECTION_FAILED:
    return "connection failed";
  case POINTRECEIVER_ERROR_TIMEOUT:
    return "timed out";
  case POINTRECEIVER_ERROR_DECODE_FAILED:
    return "decode failed";
  case POINTRECEIVER_ERROR_OUT_OF_RANGE:
    return "index out of range";
  case POINTRECEIVER_ERROR_OUT_OF_MEMORY:
    return "out of memory";
  case POINTRECEIVER_ERROR_INTERNAL:
    return "internal error";
  }
  return "unknown status";
}

pointreceiver_context *pointreceiver_create_context() {
  try {
    return new pointreceiver_context();
  } catch (const std::exception &error) {
    pc::logger()->error("pointreceiver_create_context: {}", error.what());
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

void pointreceiver_destroy_context(pointreceiver_context *ctx) {
  if (!ctx) return;
  exception_boundary("pointreceiver_destroy_context", [&] {
    delete ctx;
    pc::logger()->trace("Pointreceiver context is destroyed");
    return POINTRECEIVER_OK;
  });
}

pointreceiver_status pointreceiver_set_client_name(pointreceiver_context *ctx,
                                                   const char *client_name) {
  if (!ctx || !client_name) return POINTRECEIVER_ERROR_INVALID_ARGUMENT;
  return exception_boundary("pointreceiver_set_client_name", [&] {
    ctx->client_name = std::string(client_name);
    return POINTRECEIVER_OK;
  });
}

pointreceiver_status
pointreceiver_start_message_receiver(pointreceiver_context *ctx,
                                     const char *pointcaster_address) {
  if (!ctx || !pointcaster_address) return POINTRECEIVER_ERROR_INVALID_ARGUMENT;
  if (ctx->message_worker.joinable())
    return POINTRECEIVER_ERROR_ALREADY_RUNNING;

  return exception_boundary("pointreceiver_start_message_receiver", [&] {
    const std::string endpoint(pointcaster_address);
    zmq::socket_t socket(pc::receiver::zmq_receiver_ctx(),
                         zmq::socket_type::sub);

    try {
      // TODO why 32 msgs? what is an actually valid number here for the high
      // watermark?
      socket.set(zmq::sockopt::rcvhwm, 32);
      socket.set(zmq::sockopt::linger, 0);
      socket.set(zmq::sockopt::rcvtimeo, 100); // ms
      socket.connect(endpoint);
    } catch (const zmq::error_t &e) {
      pc::logger()->error("Message receiver failed to connect to '{}' - {}",
                          endpoint, e.what());
      return POINTRECEIVER_ERROR_CONNECTION_FAILED;
    }

    pc::logger()->info("Messaging socket open at {}", endpoint);

    ctx->message_worker = std::jthread(
        [ctx, socket = std::move(socket)](std::stop_token stop_token) mutable {
          exception_boundary("Message receive thread", [&] {
            pc::receiver::message_receive_loop(std::move(stop_token), *ctx,
                                               std::move(socket));
            return POINTRECEIVER_OK;
          });
        });
    return POINTRECEIVER_OK;
  });
}

pointreceiver_status
pointreceiver_stop_message_receiver(pointreceiver_context *ctx) {
  if (!ctx) return POINTRECEIVER_ERROR_INVALID_ARGUMENT;
  if (!ctx->message_worker.joinable()) return POINTRECEIVER_ERROR_NOT_RUNNING;

  return exception_boundary("pointreceiver_stop_message_receiver", [&] {
    pc::logger()->trace("Stopping message receiver");
    ctx->message_worker = {};
    pc::logger()->info("Message receiver thread ended");
    return POINTRECEIVER_OK;
  });
}

bool pointreceiver_message_receiver_running(pointreceiver_context *ctx) {
  return ctx && ctx->message_worker.joinable();
}

pointreceiver_status
pointreceiver_dequeue_message(pointreceiver_context *ctx,
                              pointreceiver_sync_message *out_message,
                              int timeout_ms) {
  using namespace pc::receiver;

  if (!ctx || !out_message) return POINTRECEIVER_ERROR_INVALID_ARGUMENT;

  return exception_boundary("pointreceiver_dequeue_message", [&] {
    // SyncMessage message;
    // if (!ctx->message_queue.wait_dequeue_timed(message,
    //                                            timeout_from(timeout_ms))) {
    //   return POINTRECEIVER_ERROR_TIMEOUT;
    // }

    // if (const auto *message_type = std::get_if<MessageType>(&message)) {
    //   out_message->message_type = convert_message_type(*message_type);
    //   out_message->id[0] = '\0';
    //   out_message->value_type = POINTRECEIVER_PARAM_VALUE_UNKNOWN;
    //   return POINTRECEIVER_OK;
    // }

    // if (const auto *param_update = std::get_if<ParameterUpdate>(&message)) {
    //   out_message->message_type = POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE;
    //   copy_id(out_message->id, param_update->id);

    //   if (const auto *float_value =
    //   std::get_if<float>(&param_update->value)) {
    //     out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT;
    //     out_message->value.float_val = *float_value;
    //   } else if (const auto *int_value =
    //   std::get_if<int>(&param_update->value)) {
    //     out_message->value_type = POINTRECEIVER_PARAM_VALUE_INT;
    //     out_message->value.int_val = *int_value;
    //   } else {
    //     out_message->value_type = POINTRECEIVER_PARAM_VALUE_UNKNOWN;
    //   }
    //   return POINTRECEIVER_OK;
    // }

    // if (const auto *endpoint_update = std::get_if<EndpointUpdate>(&message))
    // {
    //   out_message->message_type = POINTRECEIVER_MSG_TYPE_ENDPOINT_UPDATE;
    //   copy_id(out_message->id, endpoint_update->id);
    //   out_message->value_type = POINTRECEIVER_PARAM_VALUE_ENDPOINT_UPDATE;
    //   out_message->value.endpoint_update_val = {
    //       .port = endpoint_update->port, .active = endpoint_update->active};
    //   return POINTRECEIVER_OK;
    // }

    return POINTRECEIVER_ERROR_DECODE_FAILED;
  });
}

pointreceiver_status
pointreceiver_start_point_receiver(pointreceiver_context *ctx,
                                   const char *pointcaster_address) {
  if (!ctx || !pointcaster_address) return POINTRECEIVER_ERROR_INVALID_ARGUMENT;
  if (ctx->pointcloud_worker.joinable()) {
    return POINTRECEIVER_ERROR_ALREADY_RUNNING;
  }

  return exception_boundary("pointreceiver_start_point_receiver", [&] {
    const std::string endpoint(pointcaster_address);
    zmq::socket_t socket(pc::receiver::zmq_receiver_ctx(),
                         zmq::socket_type::sub);

    try {
      // TODO why 32 msgs? what is an actually valid number here for the
      // high watermark?
      socket.set(zmq::sockopt::rcvhwm, 32);
      socket.set(zmq::sockopt::linger, 0);
      socket.set(zmq::sockopt::rcvtimeo, 100); // ms
      socket.connect(endpoint);
    } catch (const zmq::error_t &e) {
      pc::logger()->error("Point receiver failed to connect to '{}' - {}",
                          endpoint, e.what());
      return POINTRECEIVER_ERROR_CONNECTION_FAILED;
    }

    pc::logger()->info("Point receiver connected to {}", endpoint);

    ctx->pointcloud_worker = std::jthread(
        [ctx, socket = std::move(socket)](std::stop_token stop_token) mutable {
          exception_boundary("Point cloud receive thread", [&] {
            pc::receiver::pointcloud_receive_loop(std::move(stop_token), *ctx,
                                                  std::move(socket));
            return POINTRECEIVER_OK;
          });
        });
    return POINTRECEIVER_OK;
  });
}

pointreceiver_status
pointreceiver_stop_point_receiver(pointreceiver_context *ctx) {
  if (!ctx) return POINTRECEIVER_ERROR_INVALID_ARGUMENT;
  if (!ctx->pointcloud_worker.joinable()) {
    return POINTRECEIVER_ERROR_NOT_RUNNING;
  }
  return exception_boundary("pointreceiver_stop_point_receiver", [&] {
    pc::logger()->trace("Stopping point receiver");
    ctx->pointcloud_worker = {};
    {
      std::lock_guard lock(ctx->stream_mutex);
      ctx->channels.clear();
    }
    pc::logger()->info("Point receiver thread ended");
    return POINTRECEIVER_OK;
  });
}

bool pointreceiver_point_receiver_running(pointreceiver_context *ctx) {
  return ctx && ctx->pointcloud_worker.joinable();
}

pointreceiver_status
pointreceiver_subscribe_to_point_cloud(pointreceiver_context *ctx,
                                       const char *address) {
  if (!ctx) return POINTRECEIVER_ERROR_INVALID_ARGUMENT;
  return exception_boundary("pointreceiver_subscribe_to_point_cloud", [&] {
    std::lock_guard lock(ctx->subscription_write_mutex);
    auto updated = std::make_shared<pc::receiver::SubscriptionSet>(
        *ctx->subscriptions.load());
    updated->insert(address ? address : "");
    ctx->subscriptions.store(std::move(updated));
    return POINTRECEIVER_OK;
  });
}

pointreceiver_status
pointreceiver_unsubscribe_from_point_cloud(pointreceiver_context *ctx,
                                           const char *address) {
  if (!ctx) return POINTRECEIVER_ERROR_INVALID_ARGUMENT;
  return exception_boundary("pointreceiver_unsubscribe_from_point_cloud", [&] {
    std::lock_guard lock(ctx->subscription_write_mutex);
    auto updated = std::make_shared<pc::receiver::SubscriptionSet>(
        *ctx->subscriptions.load());
    updated->erase(address ? address : "");
    ctx->subscriptions.store(std::move(updated));
    return POINTRECEIVER_OK;
  });
}

pointreceiver_status pointreceiver_dequeue_point_cloud(
    pointreceiver_context *ctx, char *out_address, size_t address_capacity,
    pointreceiver_pointcloud_frame *out_frame, int timeout_ms) {
  using namespace pc::receiver;

  if (!ctx || !out_frame || !out_address || address_capacity <= 0) {
    return POINTRECEIVER_ERROR_INVALID_ARGUMENT;
  }

  return exception_boundary("pointreceiver_dequeue_point_cloud", [&] {
    std::unique_lock lock(ctx->stream_mutex);

    auto oldest_pending = ctx->channels.end();

    ctx->stream_cv.wait_for(lock, milliseconds(timeout_ms), [&] {
      auto pending_messages =
          ctx->channels | std::views::filter([](const auto &channel) {
            return channel.second.pending;
          });
      const auto oldest = std::ranges::min_element(
          pending_messages, {},
          [](const auto &channel) { return channel.second.timestamp; });
      if (oldest == pending_messages.end()) return false;
      oldest_pending = oldest.base();
      return true;
    });

    if (oldest_pending == ctx->channels.end()) {
      return POINTRECEIVER_ERROR_TIMEOUT;
    }

    ChannelFrame &channel = oldest_pending->second;
    channel.pending = false;
    const zmq::message_t incoming_msg = std::move(channel.message);
    copy_to_buffer(out_address, address_capacity, oldest_pending->first);

    lock.unlock();

    const auto *data = static_cast<const std::byte *>(incoming_msg.data());
    const auto size = incoming_msg.size();
    const auto *separator =
        static_cast<const std::byte *>(std::memchr(data, 0, size));
    if (!separator) return POINTRECEIVER_ERROR_DECODE_FAILED;
    const std::span<const std::byte> payload(separator + 1, data + size);

    std::shared_ptr<pc::PointCloud> cloud;
    try {
      cloud = std::make_shared<pc::PointCloud>(
          pc::PointCloud::deserialize(payload));
    } catch (const std::exception &e) {
      pc::logger()->warn("dequeue deserialize threw: {} (size={})", e.what(),
                         size);
      return POINTRECEIVER_ERROR_DECODE_FAILED;
    }

    channel.cloud = cloud;
    out_frame->point_count = cloud->size();
    out_frame->positions = reinterpret_cast<const pointreceiver_position_t *>(
        cloud->positions.data());
    out_frame->colours =
        reinterpret_cast<const pointreceiver_color_t *>(cloud->colors.data());
    return POINTRECEIVER_OK;
  });
}

size_t pointreceiver_known_stream_address_count(pointreceiver_context *ctx) {
  if (!ctx) return 0;
  std::lock_guard lock(ctx->stream_mutex);
  return ctx->known_stream_addresses.size();
}

pointreceiver_status
pointreceiver_get_known_stream_address(pointreceiver_context *ctx, size_t index,
                                       char *out, size_t out_capacity) {
  if (!ctx || !out || out_capacity == 0)
    return POINTRECEIVER_ERROR_INVALID_ARGUMENT;
  return exception_boundary("pointreceiver_get_known_stream_address", [&] {
    std::lock_guard lock(ctx->stream_mutex);
    if (index >= ctx->known_stream_addresses.size()) {
      return POINTRECEIVER_ERROR_OUT_OF_RANGE;
    }
    const auto it = std::next(ctx->known_stream_addresses.begin(),
                              static_cast<std::ptrdiff_t>(index));
    if (!pc::receiver::copy_to_buffer(out, out_capacity, *it))
      return POINTRECEIVER_ERROR_INVALID_ARGUMENT;
    return POINTRECEIVER_OK;
  });
}
}

#ifndef __ANDROID__

void testMessageLoop(pointreceiver_context *ctx) {
  int count = 0;
  while (count++ < 1000) {
    pointreceiver_sync_message msg;
    // dequeue with a 5-millisecond timeout.
    if (pointreceiver_dequeue_message(ctx, &msg, 5) == POINTRECEIVER_OK) {
      pc::logger()->info("Dequeued incoming message");
      pc::logger()->info("-> {}", std::string(msg.id));
      if (msg.message_type == POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE) {
        pc::logger()->info("Parameter Update received");
        switch (msg.value_type) {
        case POINTRECEIVER_PARAM_VALUE_FLOAT:
          pc::logger()->info("Value: {}", msg.value.float_val);
          break;
        case POINTRECEIVER_PARAM_VALUE_INT:
          pc::logger()->info("Value: {}", msg.value.int_val);
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT2:
          pc::logger()->info("Value: ({}, {})", msg.value.float2_val.x,
                             msg.value.float2_val.y);
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT3:
          pc::logger()->info("Value: ({}, {}, {})", msg.value.float3_val.x,
                             msg.value.float3_val.y, msg.value.float3_val.z);
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT4:
          pc::logger()->info("Value: ({}, {}, {}, {})", msg.value.float4_val.x,
                             msg.value.float4_val.y, msg.value.float4_val.z,
                             msg.value.float4_val.w);
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT2LIST:
          pc::logger()->info("Got a Float2List of {}",
                             msg.value.float2_list_val.count);
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT3LIST:
          pc::logger()->info("Got a Float3List of {}",
                             msg.value.float3_list_val.count);
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT4LIST:
          pc::logger()->info("Got a Float4List of {}",
                             msg.value.float4_list_val.count);
          break;
        case POINTRECEIVER_PARAM_VALUE_AABBLIST:
          pc::logger()->info("Got an AABBList of {}",
                             msg.value.aabb_list_val.count);
          break;
        case POINTRECEIVER_PARAM_VALUE_CONTOURSLIST:
          pc::logger()->info("Got contours: {}",
                             msg.value.contours_list_val.count);
          break;
        default:
          pc::logger()->info("Unknown parameter update type");
          break;
        }
      } else if (msg.message_type == POINTRECEIVER_MSG_TYPE_ENDPOINT_UPDATE) {
        pc::logger()->info("Endpoint at port {} from source '{}' is {}",
                           msg.value.endpoint_update_val.port, msg.id,
                           msg.value.endpoint_update_val.active ? "active"
                                                                : "inactive");
      } else {
        pc::logger()->info("Received message type: {}",
                           static_cast<int>(msg.message_type));
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

void test_pointcloud_loop(pointreceiver_context *ctx) {
  int i = 0;
  pointreceiver_pointcloud_frame pointcloud;
  char source_id[64];
  while (i++ < 6000) {
    if (pointreceiver_dequeue_point_cloud(ctx, source_id, sizeof(source_id),
                                          &pointcloud, 5) == POINTRECEIVER_OK) {
      pc::logger()->info("--{} live points received from '{}'",
                         pointcloud.point_count, source_id);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
  }
}

int main(int argc, char *argv[]) {

  pc::logger()->set_level(spdlog::level::level_enum::trace);

  auto *ctx = pointreceiver_create_context();
  pointreceiver_set_client_name(ctx, "test_application");

  const auto *pointcloud_endpoint = argc < 2 ? "tcp://127.0.0.1:9992" : argv[1];
  const auto start_status =
      pointreceiver_start_point_receiver(ctx, pointcloud_endpoint);
  if (start_status != POINTRECEIVER_OK) {
    pc::logger()->error("Could not start point receiver on '{}': {}",
                        pointcloud_endpoint,
                        pointreceiver_status_string(start_status));
    pointreceiver_destroy_context(ctx);
    return 1;
  }

  // subscribe to all using nullptr
  pointreceiver_subscribe_to_point_cloud(ctx, nullptr);
  // pointreceiver_subscribe_to_point_cloud(ctx, "session_1");

  {
    auto pc_loop = std::jthread([&] {
      for (int i = 0; i < 3; i++) {
        test_pointcloud_loop(ctx);
      }
    });
    auto sleep = std::jthread([&] {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(5s);
    });
  }

  pointreceiver_stop_point_receiver(ctx);
  pointreceiver_destroy_context(ctx);
  return 0;
}

#endif
