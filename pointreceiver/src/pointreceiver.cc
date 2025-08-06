#include "../include/pointreceiver.h"
#include "../../src/client_sync/updates.h"

#include <chrono>
#include <concepts>
#include <cstring>
#include <execution>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <readerwriterqueue/readerwritercircularbuffer.h>
#include <readerwriterqueue/readerwriterqueue.h>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
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
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#endif

static constexpr int pointcloud_stream_port = 9992;
static constexpr int message_stream_port = 9002;

using bob::types::PointCloud;
using pc::client_sync::MessageType;
using pc::client_sync::SyncMessage;

std::unique_ptr<zmq::context_t>& zmq_ctx() {
  static auto ctx = std::make_unique<zmq::context_t>();
  return ctx;
}

struct pointreceiver_context {
  std::optional<std::string> client_name;

  std::unique_ptr<std::thread> message_thread;
  std::atomic<bool> message_thread_stop_flag{false};
  moodycamel::BlockingReaderWriterQueue<SyncMessage> message_queue;

  std::unique_ptr<std::thread> pointcloud_thread;
  std::atomic<bool> pointcloud_thread_stop_flag{false};
  moodycamel::BlockingReaderWriterCircularBuffer<PointCloud> pointcloud_queue;
  PointCloud latest_point_cloud{};

  pointreceiver_context() : message_queue(), pointcloud_queue(1) {}
};

static pointreceiver_message_type convert_message_type(MessageType message_type) {
  switch (message_type) {
  case MessageType::Connected: return POINTRECEIVER_MSG_TYPE_CONNECTED;
  case MessageType::ClientHeartbeat:
    return POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT;
  case MessageType::ClientHeartbeatResponse:
    return POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT_RESPONSE;
  case MessageType::ParameterUpdate:
    return POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE;
  case MessageType::ParameterRequest:
    return POINTRECEIVER_MSG_TYPE_PARAMETER_REQUEST;
  default: return POINTRECEIVER_MSG_TYPE_UNKNOWN;
  }
}

#ifdef __ANDROID__
void log(std::string_view arg) {
  /*auto text_cstr = std::to_string(arg).c_str();*/
  /*int (*alias)(int, const char *, const char *, ...) = __android_log_print;*/
  /*alias(ANDROID_LOG_VERBOSE, APPNAME, text_cstr);*/
  std::cout << arg << "\n";
}


  static std::ofstream file("C:/tmp/pointreceiver/log.txt", std::ios::app);
  static std::mutex file_access;
  if (file) {
    std::lock_guard<std::mutex> lock(file_access);
    file << message << "\n";
  }

#else
template <typename... Args>
void log(fmt::format_string<Args...> fmt, Args &&...args) {

  static auto logger = [] {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        "logs/pointreceiver.txt", true);
    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>("native_log", sinks.begin(),
                                                   sinks.end());
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::info);
    spdlog::register_logger(logger);
    return logger;
  }();

  logger->info(fmt, std::forward<Args>(args)...);
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
  if (auto& zmq_context = zmq_ctx()) {
    zmq_context->shutdown();
    zmq_context.reset();
  }
  log("Pointreceiver context is destroyed");
  delete ctx;
}

void pointreceiver_set_client_name(pointreceiver_context *ctx,
                                   const char *client_name) {
  if (ctx) ctx->client_name = std::string(client_name);
}

int pointreceiver_start_message_receiver(pointreceiver_context *ctx,
                                         const char *pointcaster_address) {
  const std::string pointcaster_address_str(pointcaster_address);
  
  ctx->message_thread_stop_flag.store(false, std::memory_order_relaxed);

  ctx->message_thread = std::make_unique<
      std::thread>([ctx, pointcaster_address_str]() {
    log("Beginning message receive thread");

    zmq::socket_t socket(*zmq_ctx(), zmq::socket_type::dealer);
    socket.set(zmq::sockopt::linger, 0);

    // TODO connection attempt should time out if requested
    // if (timeout_ms != 0)
    //   socket.set(zmq::sockopt::connect_timeout, timeout_ms);

    constexpr auto recv_timeout_ms = 100;
    socket.set(zmq::sockopt::rcvtimeo, recv_timeout_ms);
    if (ctx->client_name.has_value()) {
      socket.set(zmq::sockopt::routing_id, ctx->client_name.value());
    } else {
      socket.set(zmq::sockopt::routing_id, "pointreceiver");
    }

    auto endpoint = std::format("tcp://{}:{}", pointcaster_address_str,
		    message_stream_port);
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
    auto time_since_heartbeat = 0ms;

    while (!ctx->message_thread_stop_flag.load(std::memory_order_relaxed)) {
      zmq::message_t incoming_msg;
      auto recv_start_time = steady_clock::now();
      // try check for incoming msg (this times out)
      auto recv_result = socket.recv(incoming_msg, zmq::recv_flags::none);
      if (recv_result) {
        // deserialize the incoming message
        auto msg_size = incoming_msg.size();
        std::vector<std::byte> buffer(msg_size);
        auto bytes_ptr = static_cast<std::byte *>(incoming_msg.data());
        buffer.assign(bytes_ptr, bytes_ptr + msg_size);

        zpp::bits::in in(buffer);
        SyncMessage message;
        if (success(in(message))) {
          bool msg_enqueue_result;
          std::visit(
              [&](auto &&sync_message) {
                using T = std::decay_t<decltype(sync_message)>;
                if constexpr (std::same_as<T, MessageType>) {
                  if (sync_message == MessageType::ClientHeartbeatResponse) {
                    // stop our heartbeat response timer as successful?
                    msg_enqueue_result = true;
                  } else {
                    msg_enqueue_result =
                        ctx->message_queue.try_enqueue(message);
                  }
                  // check for heartbeat type
                  // pass other message types to our queue
                } else if constexpr (std::same_as<
                                         T, pc::client_sync::ParameterUpdate>) {
                  // pass all parameter updates to our queue
                  msg_enqueue_result = ctx->message_queue.try_enqueue(message);
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

      time_since_heartbeat +=
          duration_cast<milliseconds>(steady_clock::now() - recv_start_time);

      if (time_since_heartbeat >= 5s) {
        socket.send(make_msg(MessageType::ClientHeartbeat),
                    zmq::send_flags::none);
        time_since_heartbeat = 0s;
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
          } else if constexpr (std::same_as<T, Float3List>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT3LIST;
            const auto &float3List = val;
            size_t count = float3List.size();
            out_message->value.float3_list_val.count = count;
            if (count > 0) {
              // allocate memory for 'count' vectors
              out_message->value.float3_list_val.data =
                  static_cast<pointreceiver_float3_t *>(
                      std::malloc(count * sizeof(pointreceiver_float3_t)));
              if (!out_message->value.float3_list_val.data) {
                out_message->value.float3_list_val.count = 0;
                set_message_success = false;
                return;
              }
              for (size_t i = 0; i < count; i++) {
                out_message->value.float3_list_val.data[i].x = float3List[i][0];
                out_message->value.float3_list_val.data[i].y = float3List[i][1];
                out_message->value.float3_list_val.data[i].z = float3List[i][2];
              }
            } else {
              out_message->value.float3_list_val.data = nullptr;
            }
          } else if constexpr (std::same_as<T, Float4List>) {

            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT4LIST;
            const auto &float4List = val;
            size_t count = float4List.size();
            out_message->value.float4_list_val.count = count;
            if (count > 0) {
              // allocate memory for 'count' vectors
              out_message->value.float4_list_val.data =
                  static_cast<pointreceiver_float4_t *>(
                      std::malloc(count * sizeof(pointreceiver_float4_t)));
              if (!out_message->value.float4_list_val.data) {
                out_message->value.float4_list_val.count = 0;
                set_message_success = false;
                return;
              }
              for (size_t i = 0; i < count; i++) {
                out_message->value.float4_list_val.data[i].x = float4List[i][0];
                out_message->value.float4_list_val.data[i].y = float4List[i][1];
                out_message->value.float4_list_val.data[i].z = float4List[i][2];
                out_message->value.float4_list_val.data[i].w = float4List[i][3];
              }
            } else {
              out_message->value.float4_list_val.data = nullptr;
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
          } else {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_UNKNOWN;
          }
        },
        param_update.value);

    if (!set_message_success) return false;

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
  if (out_message->value_type == POINTRECEIVER_PARAM_VALUE_FLOAT3LIST) {
    if (out_message->value.float3_list_val.data) {
      free(out_message->value.float3_list_val.data);
      out_message->value.float3_list_val.data = nullptr;
    }
    out_message->value.float3_list_val.count = 0;
  } else if (out_message->value_type == POINTRECEIVER_PARAM_VALUE_FLOAT4LIST) {
    if (out_message->value.float4_list_val.data) {
      free(out_message->value.float4_list_val.data);
      out_message->value.float4_list_val.data = nullptr;
    }
    out_message->value.float3_list_val.count = 0;
  } else if (out_message->value_type == POINTRECEIVER_PARAM_VALUE_AABBLIST) {
    if (out_message->value.aabb_list_val.data) {
      free(out_message->value.aabb_list_val.data);
      out_message->value.aabb_list_val.data = nullptr;
    }
    out_message->value.aabb_list_val.count = 0;
  }

  // For other message types, no cleanup is necessary.
  return true;
}

// Start a thread that handles networking for the point cloud
int pointreceiver_start_point_receiver(pointreceiver_context *ctx,
                                       const char *pointcaster_address) {

  const std::string pointcaster_address_str(pointcaster_address);

  ctx->pointcloud_thread_stop_flag.store(false, std::memory_order_relaxed);

  ctx->pointcloud_thread = std::make_unique<std::thread>(
      [ctx, pointcaster_address_str]() {
        using namespace std::chrono;
        using namespace std::chrono_literals;

        log("Beginning pointcloud receive thread");

        // create the dish that receives point clouds
        zmq::socket_t socket(*zmq_ctx(), zmq::socket_type::dish);

        // TODO something in the set calls here is crashing Unity

        // don't retain frames in memory
        // socket.set(zmq::sockopt::linger, 0);
        // TODO what should the high watermark be... 3 frames why?
        // socket.set(zmq::sockopt::rcvhwm, 3);

        // connection attempt should time out if requested
        // if (timeout_ms != 0)
        //   socket.set(zmq::sockopt::connect_timeout, timeout_ms);

        constexpr auto recv_timeout_ms = 100;
        socket.set(zmq::sockopt::rcvtimeo, recv_timeout_ms);

        auto endpoint = std::format("tcp://{}:{}", pointcaster_address_str,
                                    pointcloud_stream_port);
        socket.connect(endpoint);
        /*log("Listening for pointcloud stream at '{}'", endpoint);*/
        log("Connecting to pointcloud stream");

        if (socket.handle() == nullptr) {
          log("Failed to connect");
          return;
        }

        socket.join("live");
        log("Connected");

        while (!ctx->pointcloud_thread_stop_flag.load(std::memory_order_relaxed)) {
          zmq::message_t incoming_msg;
          auto recv_result = socket.recv(incoming_msg, zmq::recv_flags::none);
          if (recv_result) {
            const auto msg_size = incoming_msg.size();
            std::vector<std::byte> buffer(msg_size);
            auto *bytes = static_cast<std::byte *>(incoming_msg.data());
            buffer.assign(bytes, bytes + msg_size);
            ctx->pointcloud_queue.try_enqueue(PointCloud::deserialize(buffer));
          }
        }

        socket.disconnect(endpoint);
        log("Disconnected");
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
    pointreceiver_context *ctx, pointreceiver_pointcloud_frame *out_frame,
    int timeout_ms) {
  if (!ctx || !out_frame) return false;

  auto& point_cloud = ctx->latest_point_cloud;

  bool result = ctx->pointcloud_queue.wait_dequeue_timed(
      point_cloud, std::chrono::milliseconds(timeout_ms));

  if (!result) return false;

  out_frame->point_count = static_cast<int>(point_cloud.size());
  out_frame->positions = point_cloud.positions.data();
  out_frame->colours = point_cloud.colors.data();

  return true;
}

// int pointreceiver_point_count() { return point_cloud.size(); }

// TODO make below compatible with C api

// bob::types::color *pointColors() { return point_cloud.colors.data(); }

// bob::types::position *pointPositions() { return point_cloud.positions.data(); }

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
//     std::transform(point_cloud.positions.begin(), point_cloud.positions.end(),
//                    shader_friendly_point_positions_data.begin(),
//                    transform_func);
//   }
// #else
//   // android version doesn't support parallel execution in std lib
//   std::transform(point_cloud.positions.begin(), point_cloud.positions.end(),
//                  shader_friendly_point_positions_data.begin(), transform_func);
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
      if (msg.message_type == POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE) {
        log("Parameter Update received");
        switch (msg.value_type) {
        case POINTRECEIVER_PARAM_VALUE_FLOAT:
          log("Value: {}", msg.value.float_val);
          break;
        case POINTRECEIVER_PARAM_VALUE_INT:
          log("Value: {}", msg.value.int_val);
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
        default: log("Unknown parameter update type"); break;
        }
      } else {
        log("Received message type: {}", (int)msg.message_type);
      }
      pointreceiver_free_sync_message(ctx, &msg);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

// void testPointcloudLoop() {
//   int i = 0;
//   while (i++ < 6000) {
//     using namespace std::chrono_literals;
//     std::this_thread::sleep_for(0.5ms);
//     if (!dequeuePointcloud(5)) continue;
//     auto count = pointCount();
//     // auto buffer = pointPositions();
//     log("--{} points recieved", count);
//     // auto o = buffer[200];
//     // log(fmt::format("x {}, y {}, z {}, p {}", o.x, o.y, o.z, o.__pad));
//   }
// }

int main(int argc, char *argv[]) {
  auto *ctx = pointreceiver_create_context();
  pointreceiver_set_client_name(ctx, "test_application");
  if (argc < 2) {
    // pointreceiver_start_point_receiver(ctx, "127.0.0.1");
    pointreceiver_start_message_receiver(ctx, "127.0.0.1");
  } else {
    // pointreceiver_start_point_receiver(ctx, argv[1]);
    // pointreceiver_start_point_receiver(argv[1], 5);
    pointreceiver_start_message_receiver(ctx, argv[1]);
  }
  // for (int i = 0; i < 3; i++) { testPointcloudLoop(); }
  for (int i = 0; i < 3; i++) { testMessageLoop(ctx); }
  // pointreceiver_stop_message_receiver(ctx);
  pointreceiver_stop_point_receiver(ctx);
  pointreceiver_destroy_context(ctx);
  return 0;
}

#endif