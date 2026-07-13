#include "../include/pointreceiver.h"

// TODO relocate types shared between pointcaster and pointreceiver
#include "../../src/client_sync/updates.h"
#include "../../src/structs.h"

#include <algorithm>
#include <chrono>
#include <concepts>
#include <concurrentqueue/blockingconcurrentqueue.h>
#include <condition_variable>
#include <cstring>
#include <execution>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <readerwriterqueue/readerwritercircularbuffer.h>
#include <readerwriterqueue/readerwriterqueue.h>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#ifndef ZMQ_BUILD_DRAFT_API
#define ZMQ_BUILD_DRAFT_API
#endif
#include <zmq.hpp>
#include <zpp_bits.h>

#ifdef __ANDROID__
#include <android/log.h>
#else
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#endif

static constexpr int pointcloud_stream_port = 9992;
static constexpr int message_stream_port = 9002;

using bob::types::PointCloud;
using pc::client_sync::MessageType;
using pc::client_sync::SyncMessage;

zmq::context_t &zmq_ctx() {
  constexpr auto zmq_io_thread_count = 1;
  static zmq::context_t ctx{zmq_io_thread_count};
  return ctx;
}

struct ChannelFrame {
  zmq::message_t message;
  std::uint64_t stamp = 0;
  bool pending = false;
  std::shared_ptr<PointCloud> cloud;
};

// StringHash provides transparent hashing so overwrites can look up by
// string_view without allocating
struct StringHash {
  using is_transparent = void;
  std::size_t operator()(std::string_view sv) const noexcept {
    return std::hash<std::string_view>{}(sv);
  }
};

struct pointreceiver_context {
  std::optional<std::string> client_name;

  std::unique_ptr<std::thread> message_thread;
  std::optional<std::string> message_endpoint;
  std::atomic<bool> message_thread_stop_flag{false};
  moodycamel::BlockingReaderWriterQueue<SyncMessage> message_queue;

  std::unique_ptr<std::thread> pointcloud_thread;
  std::optional<std::string> pointcloud_endpoint;
  std::atomic<bool> pointcloud_thread_stop_flag{false};

  std::mutex subscription_mutex;
  std::unordered_set<std::string> desired_subscriptions;
  bool subscriptions_dirty{true};

  std::mutex stream_mutex;
  std::condition_variable stream_cv;
  std::unordered_map<std::string, ChannelFrame, StringHash, std::equal_to<>>
      channels;
  std::uint64_t stream_stamp = 0;
  std::set<std::string> known_addresses;

  pointreceiver_context() = default;
};

static pointreceiver_message_type
convert_message_type(MessageType message_type) {
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

#ifdef __ANDROID__
template <typename... Args> void log(std::string_view arg, Args &&...args) {
  // auto text_cstr = arg.data();
  // int (*alias)(int, const char *, const char *, ...) = __android_log_print;
  // alias(ANDROID_LOG_INFO, APPNAME, text_cstr);
  // __android_log_print(ANDROID_LOG_INFO, "pointreceiver", "this is a log");
  std::cout << arg << "\n";
}

// static std::ofstream file("C:/tmp/pointreceiver/log.txt", std::ios::app);
// static std::mutex file_access;
// if (file) {
//   std::lock_guard<std::mutex> lock(file_access);
//   file << message << "\n";
// }

#else
template <typename... Args>
void log(fmt::format_string<Args...> fmt, Args &&...args) {
  spdlog::info(fmt, std::forward<Args>(args)...);
  /*static auto logger = [] {*/
  /*  auto console_sink =*/
  /*  std::make_shared<spdlog::sinks::stdout_color_sink_mt>(); auto file_sink
   * =*/
  /*  std::make_shared<spdlog::sinks::basic_file_sink_mt>(*/
  /*      "logs/pointreceiver.txt", true);*/
  /*  std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};*/
  /*  auto logger = std::make_shared<spdlog::logger>("native_log",*/
  /*  sinks.begin(),*/
  /*                                                 sinks.end());*/
  /*  logger->set_level(spdlog::level::info);*/
  /*  logger->flush_on(spdlog::level::info);*/
  /*  spdlog::register_logger(logger);*/
  /*  return logger;*/
  /*}();*/
  /**/
  /*logger->info(fmt, std::forward<Args>(args)...);*/
}
#endif

extern "C" {

pointreceiver_context *pointreceiver_create_context() {
  return new pointreceiver_context();
}

void pointreceiver_destroy_context(pointreceiver_context *ctx) {
  if (ctx) {
    if (ctx->message_thread) pointreceiver_stop_message_receiver(ctx);
    if (ctx->pointcloud_thread) pointreceiver_stop_point_receiver(ctx);
  }
  log("Pointreceiver context is destroyed");
  delete ctx;
  ctx = nullptr;
}

void pointreceiver_set_client_name(pointreceiver_context *ctx,
                                   const char *client_name) {
  if (ctx) ctx->client_name = std::string(client_name);
}

int pointreceiver_start_message_receiver(pointreceiver_context *ctx,
                                         const char *pointcaster_address) {

  ctx->message_thread_stop_flag.store(false, std::memory_order_relaxed);

  const std::string endpoint(pointcaster_address);
  ctx->message_endpoint = endpoint;
  ctx->message_thread = std::make_unique<std::thread>([ctx, endpoint]() {
    log("Beginning message receive thread");

    zmq::socket_t socket(zmq_ctx(), zmq::socket_type::dealer);
    socket.set(zmq::sockopt::linger, 0);

    // TODO connection attempt should time out if requested
    // if (timeout_ms != 0)
    //   socket.set(zmq::sockopt::connect_timeout, timeout_ms);

    constexpr auto recv_timeout_ms = 100;
    socket.set(zmq::sockopt::rcvtimeo, recv_timeout_ms);
    socket.set(zmq::sockopt::routing_id,
               ctx->client_name.value_or("pointreceiver"));
    socket.connect(endpoint);

    if (socket.handle() == nullptr) {
      log("Failed to connect");
      return;
    }
    log("Messaging socket open");

    // convenience map contains our message type signals pre-populated as
    // serialized buffers
    static const std::unordered_map<MessageType, std::vector<std::byte>>
        msg_type_buffers = []() {
          std::unordered_map<MessageType, std::vector<std::byte>> map;
          const auto add = [&map](MessageType type) {
            auto [buffer, out] = zpp::bits::data_out();
            auto result = out(SyncMessage{type});
            if (success(result)) {
              map.emplace(type, std::move(buffer));
            } else {
              throw std::runtime_error("Failed to serialize MessageType");
            }
          };
          add(MessageType::Connected);
          add(MessageType::ClientHeartbeat);
          add(MessageType::ParameterRequest);
          // Add additional cases as needed.
          return map;
        }();
    // along with the following convenience func we can easily get our
    // MessageType as a zmq::message_t ready to send
    static auto make_msg = [](MessageType type) -> zmq::message_t {
      const auto &buffer = msg_type_buffers.at(type);
      return zmq::message_t(buffer.data(), buffer.size());
    };

    // Send our server a connected message
    socket.send(make_msg(MessageType::Connected), zmq::send_flags::none);

    // Request all publishing parameters
    socket.send(make_msg(MessageType::ParameterRequest), zmq::send_flags::none);

    using namespace std::chrono;
    using namespace std::chrono_literals;

    auto last_heartbeat_send = steady_clock::now() - 5s;

    zmq::pollitem_t socket_poll_items[] = {{socket.handle(), 0, ZMQ_POLLIN, 0}};
    zmq::message_t incoming_msg;

    while (!ctx->message_thread_stop_flag.load(std::memory_order_relaxed)) {
      zmq::poll(socket_poll_items, 1,
                std::chrono::milliseconds(recv_timeout_ms));

      if (socket_poll_items[0].revents & ZMQ_POLLIN) {
        auto recv_result = socket.recv(incoming_msg, zmq::recv_flags::none);
        if (recv_result) {
          const auto msg_size = incoming_msg.size();
          std::vector<std::byte> buffer(msg_size);
          auto *bytes_ptr = static_cast<std::byte *>(incoming_msg.data());
          buffer.assign(bytes_ptr, bytes_ptr + msg_size);

          zpp::bits::in in(buffer);
          SyncMessage message;
          if (success(in(message))) {
            bool msg_enqueue_result = false;
            std::visit(
                [&](auto &&sync_message) {
                  using T = std::decay_t<decltype(sync_message)>;
                  if constexpr (std::same_as<T, MessageType> ||
                                std::same_as<
                                    T, pc::client_sync::ParameterUpdate> ||
                                std::same_as<T,
                                             pc::client_sync::EndpointUpdate>) {
                    msg_enqueue_result =
                        ctx->message_queue.try_enqueue(message);
                  } else {
                    msg_enqueue_result = false;
                    log("Couldn't enqueue message type");
                  }
                },
                message);

            if (!msg_enqueue_result) {
              log("Failed to enqueue incoming message from Pointcaster");
            }
          } else {
            log("Failed to deserialise incoming message from Pointcaster");
          }
        }
      }

      const auto now = steady_clock::now();

      if (now - last_heartbeat_send >= 5s) {
        socket.send(make_msg(MessageType::ClientHeartbeat),
                    zmq::send_flags::none);
        last_heartbeat_send = now;
      }
    }
    log("Received thread stop flag");

    socket.disconnect(endpoint);
    log("Disconnected from socket");
  });

  return 0;
}

int pointreceiver_stop_message_receiver(pointreceiver_context *ctx) {
  if (ctx->message_thread) {
    log("Stopping message receiver");
    ctx->message_thread_stop_flag.store(true);
    ctx->message_thread->join();
    log("Message receiver thread ended");
    ctx->message_thread.reset();
  }
  return 0;
}

bool pointreceiver_dequeue_message(pointreceiver_context *ctx,
                                   pointreceiver_sync_message *out_message,
                                   int timeout_ms) {
  if (!ctx || !out_message) return false;

  using namespace pc::client_sync;
  using namespace std::chrono;
  using namespace std::chrono_literals;

  pc::client_sync::SyncMessage message;
  bool result =
      ctx->message_queue.wait_dequeue_timed(message, milliseconds(timeout_ms));
  if (!result) return false;

  if (std::holds_alternative<MessageType>(message)) {
    auto message_type = std::get<MessageType>(message);
    out_message->message_type = convert_message_type(message_type);
    out_message->id[0] = '\0';
    out_message->value_type = POINTRECEIVER_PARAM_VALUE_UNKNOWN;
  } else if (std::holds_alternative<ParameterUpdate>(message)) {
    const auto &param_update = std::get<ParameterUpdate>(message);
    out_message->message_type = POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE;
    // Copy the id (ensure we don't overflow our fixed-size buffer)
    std::strncpy(out_message->id, param_update.id.c_str(), 256);
    out_message->id[255] = '\0';

    bool set_message_success = true;
    // set the type of the update value.
    std::visit(
        [&](const auto &val) {
          using T = std::decay_t<decltype(val)>;
          if constexpr (std::same_as<T, float>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT;
            out_message->value.float_val = val;
          } else if constexpr (std::same_as<T, int>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_INT;
            out_message->value.int_val = val;
          } else if constexpr (std::same_as<T, pc::types::Float2> ||
                               std::same_as<T, std::array<float, 2>>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT2;
            out_message->value.float2_val.x = val[0];
            out_message->value.float2_val.y = val[1];
          } else if constexpr (std::same_as<T, pc::types::Float3> ||
                               std::same_as<T, std::array<float, 3>>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT3;
            out_message->value.float3_val.x = val[0];
            out_message->value.float3_val.y = val[1];
            out_message->value.float3_val.z = val[2];
          } else if constexpr (std::same_as<T, pc::types::Float4> ||
                               std::same_as<T, std::array<float, 4>>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT4;
            out_message->value.float4_val.x = val[0];
            out_message->value.float4_val.y = val[1];
            out_message->value.float4_val.z = val[2];
            out_message->value.float4_val.w = val[3];
          } else if constexpr (std::same_as<T, Float2List> ||
                               std::same_as<T,
                                            std::list<std::array<float, 2>>>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT2LIST;
            const auto &float2List = val;
            const size_t count = float2List.size();
            out_message->value.float2_list_val.count = count;
            if (count == 0) {
              out_message->value.float2_list_val.data = nullptr;
              return;
            }
            out_message->value.float2_list_val.data =
                static_cast<pointreceiver_float2_t *>(
                    std::malloc(count * sizeof(pointreceiver_float2_t)));
            if (!out_message->value.float2_list_val.data) {
              out_message->value.float2_list_val.count = 0;
              set_message_success = false;
              return;
            }
            size_t i = 0;
            for (const auto &v : float2List) {
              out_message->value.float2_list_val.data[i].x = v[0];
              out_message->value.float2_list_val.data[i].y = v[1];
              ++i;
            }
          } else if constexpr (std::same_as<T, Float3List> ||
                               std::same_as<T,
                                            std::list<std::array<float, 3>>>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT3LIST;
            const auto &float3List = val;
            const size_t count = float3List.size();
            out_message->value.float3_list_val.count = count;
            if (count == 0) {
              out_message->value.float3_list_val.data = nullptr;
              return;
            }
            out_message->value.float3_list_val.data =
                static_cast<pointreceiver_float3_t *>(
                    std::malloc(count * sizeof(pointreceiver_float3_t)));
            if (!out_message->value.float3_list_val.data) {
              out_message->value.float3_list_val.count = 0;
              set_message_success = false;
              return;
            }
            size_t i = 0;
            for (const auto &v : float3List) {
              out_message->value.float3_list_val.data[i].x = v[0];
              out_message->value.float3_list_val.data[i].y = v[1];
              out_message->value.float3_list_val.data[i].z = v[2];
              ++i;
            }
          } else if constexpr (std::same_as<T, Float4List> ||
                               std::same_as<T,
                                            std::list<std::array<float, 4>>>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT4LIST;
            const auto &float4List = val;
            const size_t count = float4List.size();
            out_message->value.float4_list_val.count = count;
            if (count == 0) {
              out_message->value.float4_list_val.data = nullptr;
              return;
            }
            out_message->value.float4_list_val.data =
                static_cast<pointreceiver_float4_t *>(
                    std::malloc(count * sizeof(pointreceiver_float4_t)));
            if (!out_message->value.float4_list_val.data) {
              out_message->value.float4_list_val.count = 0;
              set_message_success = false;
              return;
            }
            size_t i = 0;
            for (const auto &v : float4List) {
              out_message->value.float4_list_val.data[i].x = v[0];
              out_message->value.float4_list_val.data[i].y = v[1];
              out_message->value.float4_list_val.data[i].z = v[2];
              out_message->value.float4_list_val.data[i].w = v[3];
              ++i;
            }

          } else if constexpr (std::same_as<T, AABBList>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_AABBLIST;
            const auto &aabbList = val;
            size_t count = aabbList.size();
            out_message->value.aabb_list_val.count = count;
            if (count > 0) {
              // allocate memory for count AABB elements.
              out_message->value.aabb_list_val.data =
                  static_cast<pointreceiver_aabb_t *>(
                      std::malloc(count * sizeof(pointreceiver_aabb_t)));
              if (!out_message->value.aabb_list_val.data) {
                out_message->value.aabb_list_val.count = 0;
                set_message_success = false;
                return;
              }
              // copy each AABB element.
              for (size_t i = 0; i < count; ++i) {
                const auto &aabb = aabbList[i];
                for (int j = 0; j < 3; ++j) {
                  out_message->value.aabb_list_val.data[i].min[j] = aabb[0][j];
                  out_message->value.aabb_list_val.data[i].max[j] = aabb[1][j];
                }
              }
            } else {
              out_message->value.aabb_list_val.data = nullptr;
            }
          } else if constexpr (std::same_as<T, ContoursList>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_CONTOURSLIST;

            const auto &contours = val;
            const size_t contour_count = contours.size();
            out_message->value.contours_list_val.count = contour_count;

            if (contour_count == 0) {
              out_message->value.contours_list_val.data = nullptr;
              return;
            }

            // allocate outer contour array
            out_message->value.contours_list_val.data =
                static_cast<pointreceiver_contour_t *>(std::malloc(
                    contour_count * sizeof(pointreceiver_contour_t)));

            if (!out_message->value.contours_list_val.data) {
              out_message->value.contours_list_val.count = 0;
              set_message_success = false;
              return;
            }
            size_t total_vertex_count = 0;
            for (size_t i = 0; i < contour_count; ++i) {
              total_vertex_count += contours[i].size();
            }
            // allocate one vertex buffer
            pointreceiver_float2_t *flat_vertices = nullptr;
            if (total_vertex_count > 0) {
              flat_vertices = static_cast<pointreceiver_float2_t *>(std::malloc(
                  total_vertex_count * sizeof(pointreceiver_float2_t)));

              if (!flat_vertices) {
                std::free(out_message->value.contours_list_val.data);
                out_message->value.contours_list_val.data = nullptr;
                out_message->value.contours_list_val.count = 0;
                set_message_success = false;
                return;
              }
            }
            // fill and assign each contour in a range of the flat buffer
            size_t write_index = 0;
            for (size_t contour_index = 0; contour_index < contour_count;
                 ++contour_index) {
              const auto &contour = contours[contour_index];
              const size_t vertex_count = contour.size();

              pointreceiver_contour_t &out_contour =
                  out_message->value.contours_list_val.data[contour_index];

              out_contour.count = vertex_count;
              out_contour.data =
                  (vertex_count > 0) ? (flat_vertices + write_index) : nullptr;

              for (size_t vertex_index = 0; vertex_index < vertex_count;
                   ++vertex_index) {
                out_contour.data[vertex_index].x = contour[vertex_index][0];
                out_contour.data[vertex_index].y = contour[vertex_index][1];
              }
              write_index += vertex_count;
            }
          }

          else {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_UNKNOWN;
          }
        },
        param_update.value);

    if (!set_message_success) return false;

  } else if (std::holds_alternative<EndpointUpdate>(message)) {
    // parse the endpoint update and ensure we have connected sockets
    auto endpoint_update = std::get<EndpointUpdate>(message);
    if (endpoint_update.active) {
      // ensure_static_socket_active(ctx, endpoint_update.id,
      // endpoint_update.port);
    } else {
      // ctx->static_pointcloud_sockets.erase(endpoint_update.id);
    }

    // pass the endpoint update to the client too in case they need it
    out_message->message_type = POINTRECEIVER_MSG_TYPE_ENDPOINT_UPDATE;
    std::strncpy(out_message->id, endpoint_update.id.c_str(),
                 endpoint_update.id.size());
    out_message->id[endpoint_update.id.size()] = '\0';
    out_message->value.endpoint_update_val = {.port = endpoint_update.port,
                                              .active = endpoint_update.active};
  } else {
    return false;
  }
  return true;
}

bool pointreceiver_free_sync_message(pointreceiver_context *ctx,
                                     pointreceiver_sync_message *out_message) {
  if (!out_message) return false;

  // if the message contains a type that uses dynamically allocated memory, we
  // must free it
  switch (out_message->value_type) {
  case POINTRECEIVER_PARAM_VALUE_FLOAT3LIST: {
    free(out_message->value.float3_list_val.data);
    out_message->value.float3_list_val.data = NULL;
    out_message->value.float3_list_val.count = 0;
    break;
  }
  case POINTRECEIVER_PARAM_VALUE_FLOAT4LIST: {
    free(out_message->value.float4_list_val.data);
    out_message->value.float4_list_val.data = NULL;
    out_message->value.float4_list_val.count = 0;
    break;
  }
  case POINTRECEIVER_PARAM_VALUE_AABBLIST: {
    free(out_message->value.aabb_list_val.data);
    out_message->value.aabb_list_val.data = NULL;
    out_message->value.aabb_list_val.count = 0;
    break;
  }
  case POINTRECEIVER_PARAM_VALUE_CONTOURSLIST: {
    pointreceiver_contours_list_t *contours_list =
        &out_message->value.contours_list_val;
    if (contours_list && contours_list->data) {
      // find the base pointer of the flat buffer (first non-empty contour)
      pointreceiver_float2_t *flat_vertices = NULL;
      for (size_t i = 0; i < contours_list->count; ++i) {
        if (contours_list->data[i].data) {
          flat_vertices = contours_list->data[i].data;
          break;
        }
      }
      if (flat_vertices) {
        free(flat_vertices);
      }
      free(contours_list->data);
      contours_list->data = NULL;
    }
    contours_list->count = 0;
    break;
  }
  default: {
    // for other message types, no cleanup necessary
    break;
  }
  }
  return true;
}

// Start a thread that handles networking for the point cloud
int pointreceiver_start_point_receiver(pointreceiver_context *ctx,
                                       const char *pointcaster_address) {

  const std::string endpoint(pointcaster_address);
  ctx->pointcloud_endpoint = endpoint;

  ctx->pointcloud_thread_stop_flag.store(false, std::memory_order_relaxed);

  ctx->pointcloud_thread = std::make_unique<std::thread>([ctx, endpoint]() {
    zmq::socket_t socket(zmq_ctx(), zmq::socket_type::sub);

    // socket.set(zmq::sockopt::rcvhwm, 8); 
    socket.set(zmq::sockopt::rcvhwm, 32); 

    socket.set(zmq::sockopt::linger, 0);
    socket.connect(endpoint);

    std::unordered_set<std::string> applied_subscriptions;
    zmq::pollitem_t items[] = {{socket.handle(), 0, ZMQ_POLLIN, 0}};

    while (!ctx->pointcloud_thread_stop_flag.load(std::memory_order_relaxed)) {
      // 1. sync socket subscriptions to desired intent
      {
        std::lock_guard lock(ctx->subscription_mutex);
        if (ctx->subscriptions_dirty) {
          for (const auto &address : ctx->desired_subscriptions)
            if (!applied_subscriptions.contains(address))
              socket.set(zmq::sockopt::subscribe,
                         address.empty() ? address : address + '\0');
          for (const auto &address : applied_subscriptions)
            if (!ctx->desired_subscriptions.contains(address))
              socket.set(zmq::sockopt::unsubscribe,
                         address.empty() ? address : address + '\0');
          applied_subscriptions = ctx->desired_subscriptions;
          ctx->subscriptions_dirty = false;
        }
      }

      zmq::poll(items, 1, std::chrono::milliseconds(20));

      // 2. drain all queued frames; stash the latest raw bytes per channel
      while (true) {
        zmq::message_t message;
        if (!socket.recv(message, zmq::recv_flags::dontwait)) break;

        const auto *data = static_cast<const std::byte *>(message.data());
        const auto size = message.size();
        const auto *separator =
            static_cast<const std::byte *>(std::memchr(data, 0, size));
        if (!separator) continue; // malformed, no separator

        const std::string_view address(
            reinterpret_cast<const char *>(data),
            static_cast<std::size_t>(separator - data));
        {
          std::lock_guard lock(ctx->stream_mutex);
          auto it =
              ctx->channels.find(address); // heterogeneous, no alloc on hit
          if (it == ctx->channels.end()) {
            it = ctx->channels.emplace(std::string(address), ChannelFrame{})
                     .first;
            ctx->known_addresses.emplace(address);
          }
          it->second.message =
              std::move(message); // overwrites + frees stale frame
          it->second.stamp = ++ctx->stream_stamp;
          it->second.pending = true;
        }
        ctx->stream_cv.notify_one();
      }
    }
  });

  return 0;
}

int pointreceiver_stop_point_receiver(pointreceiver_context *ctx) {
  ctx->pointcloud_thread_stop_flag.store(true, std::memory_order_relaxed);
  auto &thread = ctx->pointcloud_thread;
  if (thread && thread->joinable()) {
    thread->join();
    thread.reset();
  }
  return 0;
}

bool pointreceiver_dequeue_point_cloud(
    pointreceiver_context *ctx, char *out_address, int address_capacity,
    pointreceiver_pointcloud_frame *out_frame, int timeout_ms) {
  if (!ctx || !out_frame) return false;

  // there may be more than one frame received from the same
  // publisher in the queue, so we grab only the freshest message
  // and deserialize that only ...
  // (since deserialisation can be expensive for large clouds)

  using namespace std::chrono;
  using namespace std::chrono_literals;

  zmq::message_t freshest;
  std::string address;
  {
    std::unique_lock lock(ctx->stream_mutex);
    const bool ready =
        ctx->stream_cv.wait_for(lock, milliseconds(timeout_ms), [&] {
          return std::ranges::any_of(ctx->channels, [](const auto &entry) {
            return entry.second.pending;
          });
        });
    if (!ready) return false;

    auto freshest_it = ctx->channels.end();
    for (auto it = ctx->channels.begin(); it != ctx->channels.end(); ++it)
      if (it->second.pending && (freshest_it == ctx->channels.end() ||
                                 it->second.stamp < freshest_it->second.stamp))
        freshest_it = it;
    if (freshest_it == ctx->channels.end()) return false;

    freshest_it->second.pending = false;
    freshest = std::move(freshest_it->second.message); // take the raw bytes
    address = freshest_it->first;                      // copy the address key
    if (out_address && address_capacity > 0)
      std::snprintf(out_address, address_capacity, "%s", address.c_str());
  }

  const auto *data = static_cast<const std::byte *>(freshest.data());
  const auto size = freshest.size();
  const auto *separator =
      static_cast<const std::byte *>(std::memchr(data, 0, size));
  if (!separator) return false;
  const std::span<const std::byte> payload(separator + 1, data + size);

  std::shared_ptr<PointCloud> cloud;
  try {
    cloud = std::make_shared<PointCloud>(PointCloud::deserialize(payload));
  } catch (const std::exception &e) {
    log("dequeue deserialize threw: {} (size={})", e.what(), size);
    return false; // caller keeps showing its last published frame for this
                  // channel
  }

  {
    std::lock_guard lock(ctx->stream_mutex);
    ctx->channels[address].cloud = cloud;
  }

  out_frame->point_count = static_cast<int>(cloud->size());
  out_frame->positions = cloud->positions.data();
  out_frame->colours = cloud->colors.data();
  return true;
}

int pointreceiver_subscribe_to_point_cloud(pointreceiver_context *ctx,
                                           const char *address) {
  if (!ctx) return -1;
  std::lock_guard lock(ctx->subscription_mutex);
  ctx->desired_subscriptions.insert(address ? address : "");
  ctx->subscriptions_dirty = true;
  return 0;
}

int pointreceiver_unsubscribe_from_point_cloud(pointreceiver_context *ctx,
                                               const char *address) {
  if (!ctx) return -1;
  std::lock_guard lock(ctx->subscription_mutex);
  ctx->desired_subscriptions.erase(address ? address : "");
  ctx->subscriptions_dirty = true;
  return 0;
}

size_t pointreceiver_known_address_count(pointreceiver_context *ctx) {
  if (!ctx) return 0;
  std::lock_guard lock(ctx->stream_mutex);
  return ctx->known_addresses.size();
}

bool pointreceiver_get_known_address(pointreceiver_context *ctx, size_t index,
                                     char *out, size_t out_cap) {
  if (!ctx || !out || out_cap == 0) return false;
  std::lock_guard lock(ctx->stream_mutex);
  if (index >= ctx->known_addresses.size()) return false;
  const auto it = std::next(ctx->known_addresses.begin(),
                            static_cast<std::ptrdiff_t>(index));
  std::snprintf(out, out_cap, "%s", it->c_str());
  return true;
}

// int pointreceiver_point_count() { return point_cloud.size(); }

// TODO make below compatible with C api

// bob::types::color *pointColors() { return point_cloud.colors.data(); }

// bob::types::position *pointPositions() { return point_cloud.positions.data();
// }

// std::vector<std::array<float, 4>> shader_friendly_point_positions_data;

// enum class OrientationType { unchanged = 0, flip_x, flip_x_z, flip_z };
// float *shaderFriendlyPointPositions(bool parallel_transform,
//                                     OrientationType orientation_type) {
//   std::size_t size = point_cloud.positions.size();
//   shader_friendly_point_positions_data.resize(size);

//   std::array<float, 3> orientation{1, 1, 1};
//   if (orientation_type == OrientationType::flip_x) {
//     orientation[0] = -1;
//   } else if (orientation_type == OrientationType::flip_z) {
//     orientation[2] = -1;
//   } else if (orientation_type == OrientationType::flip_x_z) {
//     orientation[0] = -1;
//     orientation[2] = -1;
//   }

//   const auto transform_func =
//       [&](const bob::types::position &pos) -> std::array<float, 4> {
//     return {static_cast<float>(pos.x) * orientation[0] / 1000.0f,
//             static_cast<float>(pos.y) * orientation[1] / 1000.0f,
//             static_cast<float>(pos.z) * orientation[2] / 1000.0f, 0.0f};
//   };

// #ifndef __ANDROID__
//   if (parallel_transform) {
//     std::transform(std::execution::par, point_cloud.positions.begin(),
//                    point_cloud.positions.end(),
//                    shader_friendly_point_positions_data.begin(),
//                    transform_func);
//   } else {
//     std::transform(point_cloud.positions.begin(),
//     point_cloud.positions.end(),
//                    shader_friendly_point_positions_data.begin(),
//                    transform_func);
//   }
// #else
//   // android version doesn't support parallel execution in std lib
//   std::transform(point_cloud.positions.begin(), point_cloud.positions.end(),
//                  shader_friendly_point_positions_data.begin(),
//                  transform_func);
// #endif

//   return shader_friendly_point_positions_data.front().data();
// }
}

#ifndef __ANDROID__

void testMessageLoop(pointreceiver_context *ctx) {
  int count = 0;
  while (count++ < 1000) {
    pointreceiver_sync_message msg;
    // dequeue with a 5-millisecond timeout.
    if (pointreceiver_dequeue_message(ctx, &msg, 5)) {
      log("Dequeued incoming message");
      log("-> {}", std::string(msg.id));
      if (msg.message_type == POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE) {
        log("Parameter Update received");
        switch (msg.value_type) {
        case POINTRECEIVER_PARAM_VALUE_FLOAT:
          log("Value: {}", msg.value.float_val);
          break;
        case POINTRECEIVER_PARAM_VALUE_INT:
          log("Value: {}", msg.value.int_val);
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT2:
          log("Value: ({}, {})", msg.value.float2_val.x,
              msg.value.float2_val.y);
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT3:
          log("Value: ({}, {}, {})", msg.value.float3_val.x,
              msg.value.float3_val.y, msg.value.float3_val.z);
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT4:
          log("Dequeued a raw float4");
          log("Value: ({}, {}, {}, {})", msg.value.float4_val.x,
              msg.value.float4_val.y, msg.value.float4_val.z,
              msg.value.float4_val.w);
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT2LIST:
          // TODO log contents or at least size?
          log("Got a Float2List here");
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT3LIST:
          // TODO log contents or at least size?
          log("Got a Float3List here");
          break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT4LIST:
          // TODO log contents or at least size?
          log("Got a Float4List here");
          break;
        case POINTRECEIVER_PARAM_VALUE_AABBLIST:
          log("Got an AABBList here");
          break;
        case POINTRECEIVER_PARAM_VALUE_CONTOURSLIST:
          log("Got contours: {}", msg.value.contours_list_val.count);
          break;
        default:
          log("Unknown parameter update type");
          break;
        }
      } else if (msg.message_type == POINTRECEIVER_MSG_TYPE_ENDPOINT_UPDATE) {
        log("Got an Endpoint update");
        auto &endpoint_update = msg.value.endpoint_update_val;
        log("Endpoint at port {} from source '{}' is {}",
            msg.value.endpoint_update_val.port, msg.id,
            msg.value.endpoint_update_val.active ? "active" : "inactive");
      } else {
        log("Received message type: {}", (int)msg.message_type);
      }
      pointreceiver_free_sync_message(ctx, &msg);
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
                                          &pointcloud, 5)) {
      log("--{} live points received from '{}'", pointcloud.point_count,
          source_id);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    /*log("static");*/
    /*if (pointreceiver_dequeue_static_point_cloud(ctx, source_id,*/
    /*&pointcloud, 5)) {*/
    /*log("--static cloud from '{}' received", source_id);*/
    /*}*/
    /*log("post static");*/
  }
}

int main(int argc, char *argv[]) {
  auto *ctx = pointreceiver_create_context();
  pointreceiver_set_client_name(ctx, "test_application");
  if (argc < 2) {
    pointreceiver_start_point_receiver(ctx, "tcp://127.0.0.1:9992");
    // pointreceiver_start_message_receiver(ctx, "tcp://127.0.0.1:9002");
  } else {
    pointreceiver_start_point_receiver(ctx, argv[1]);
    // pointreceiver_start_message_receiver(ctx, argv[2]);
  }
  // subscribe to all using nullptr
  pointreceiver_subscribe_to_point_cloud(ctx, nullptr);

  {
    /*std::this_thread::sleep_for(std::chrono::seconds(5));*/
    auto pc_loop = std::jthread([&] {
      for (int i = 0; i < 3; i++) {
        test_pointcloud_loop(ctx);
      }
    });
    // auto msg_loop = std::jthread([&] {
    //   for (int i = 0; i < 3; i++) { testMessageLoop(ctx); }
    // });
    auto sleep = std::jthread([&] {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(5s);
    });
  }
  // pointreceiver_stop_message_receiver(ctx);
  pointreceiver_stop_point_receiver(ctx);
  pointreceiver_destroy_context(ctx);
  return 0;
}

#endif
