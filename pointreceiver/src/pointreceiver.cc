#include "../include/pointreceiver.h"
#include "../../src/client_sync/updates.h"

#include <chrono>
#include <cstring>
#include <execution>
#include <iostream>
#include <mutex>
#include <numeric>
#include <readerwriterqueue/readerwritercircularbuffer.h>
#include <readerwriterqueue/readerwriterqueue.h>
#include <thread>
#include <variant>
#include <vector>
#define ZMQ_BUILD_DRAFT_API
#include <zmq.hpp>
#include <zpp_bits.h>

#ifdef __ANDROID__
#include <android/log.h>
#else
#include <spdlog/spdlog.h>
#endif


static constexpr int pointcloud_stream_port = 9992;
static constexpr int message_stream_port = 9002;

using bob::types::PointCloud;
using pc::client_sync::SyncMessage;

static std::unique_ptr<zmq::context_t> zmq_ctx;
static std::once_flag zmq_ctx_once;

static moodycamel::BlockingReaderWriterCircularBuffer<PointCloud>
    cloud_queue(1);

std::unique_ptr<std::jthread> pointcloud_thread;

struct pointreceiver_context {
  moodycamel::BlockingReaderWriterQueue<SyncMessage> message_queue;
  std::unique_ptr<std::jthread> message_thread;

  pointreceiver_context() : message_queue() {}
};

static pointreceiver_message_type
convertMessageType(pc::client_sync::MessageType message_type) {
  switch (message_type) {
  case pc::client_sync::MessageType::Connected: return POINTRECEIVER_MSG_TYPE_CONNECTED;
  case pc::client_sync::MessageType::ClientHeartbeat:
    return POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT;
  case pc::client_sync::MessageType::ClientHeartbeatResponse:
    return POINTRECEIVER_MSG_TYPE_CLIENT_HEARTBEAT_RESPONSE;
  case pc::client_sync::MessageType::ParameterUpdate:
    return POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE;
  case pc::client_sync::MessageType::ParameterRequest:
    return POINTRECEIVER_MSG_TYPE_PARAMETER_REQUEST;
  default: return POINTRECEIVER_MSG_TYPE_UNKNOWN;
  }
}

#ifdef __ANDROID__
void log(std::string text) {
  auto text_cstr = text.c_str();
  int (*alias)(int, const char *, const char *, ...) = __android_log_print;
  alias(ANDROID_LOG_VERBOSE, APPNAME, text_cstr);
}
#else
template <typename... Args>
void log(fmt::format_string<Args...> fmt, Args &&...args) {
  spdlog::info(fmt, std::forward<Args>(args)...);
}
#endif

extern "C" {

pointreceiver_context *pointreceiver_create_context() {
  return new pointreceiver_context();
}

void pointreceiver_destroy_context(pointreceiver_context *ctx) {
  if (ctx) {
    if (ctx->message_thread) { ctx->message_thread->request_stop(); }
  }
  delete ctx;
}

int pointreceiver_start_message_receiver(pointreceiver_context *ctx,
                                         const char *pointcaster_address) {
  // initialise the zmq context if it hasn't already been initialised
  std::call_once(zmq_ctx_once,
                 [] { zmq_ctx = std::make_unique<zmq::context_t>(); });

  const std::string pointcaster_address_str(pointcaster_address);

  ctx->message_thread = std::make_unique<std::jthread>(
      [ctx, pointcaster_address_str](std::stop_token stop_token) {
        log("Beginning message receive thread");

        zmq::socket_t socket(*zmq_ctx, zmq::socket_type::dealer);
        socket.set(zmq::sockopt::linger, 0);

        // TODO connection attempt should time out if requested
        // if (timeout_ms != 0)
        //   socket.set(zmq::sockopt::connect_timeout, timeout_ms);

        constexpr auto recv_timeout_ms = 100;
        socket.set(zmq::sockopt::rcvtimeo, recv_timeout_ms);
        // TODO what should the id be??
        // socket.set(zmq::sockopt::routing_id, "dealer_socket");

        auto endpoint = fmt::format("tcp://{}:{}", pointcaster_address_str,
                                    message_stream_port);
        socket.connect(endpoint);

        if (socket.handle() == nullptr) {
          log("Failed to connect");
          return;
        }

        // Send our server a connected message
        constexpr auto connected_signal = pc::client_sync::MessageType::Connected;
        zmq::message_t msg(&connected_signal, sizeof(connected_signal));
        socket.send(msg, zmq::send_flags::none);

        log("Listening for messages at '{}'", endpoint);

        while (!stop_token.stop_requested()) {
          zmq::message_t incoming_msg;
          auto recv_result = socket.recv(incoming_msg, zmq::recv_flags::none);
          if (!recv_result) continue;

          // deserialize the incoming message
          auto msg_size = incoming_msg.size();
          std::vector<std::byte> buffer(msg_size);
          auto bytes_ptr = static_cast<std::byte *>(incoming_msg.data());
          buffer.assign(bytes_ptr, bytes_ptr + msg_size);

          zpp::bits::in in(buffer);
          SyncMessage message;
          if (failure(in(message))) {
            log("Failed to deserialise incoming message from Pointcaster");
            continue;
          }
          if (!ctx->message_queue.try_enqueue(message)) {
            log("Failed to enqueue incoming message from Pointcaster");
          }
        }

        socket.disconnect(endpoint);
        log("Disconnected");
      });

  return 0;
}

int pointreceiver_stop_message_receiver(pointreceiver_context *ctx) {
  if (ctx->message_thread) {
    ctx->message_thread->request_stop();
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

  if (std::holds_alternative<pc::client_sync::MessageType>(message)) {
    auto message_type = std::get<pc::client_sync::MessageType>(message);
    out_message->message_type = convertMessageType(message_type);
    out_message->id[0] = '\0';
    out_message->value_type = POINTRECEIVER_PARAM_VALUE_UNKNOWN;
  } else if (std::holds_alternative<ParameterUpdate>(message)) {
    const auto &param_update = std::get<ParameterUpdate>(message);
    out_message->message_type = POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE;
    // Copy the id (ensure we don't overflow our fixed-size buffer)
    std::strncpy(out_message->id, param_update.id.c_str(), 256);
    out_message->id[255] = '\0';

    // set the type of the update value.
    std::visit(
        [&](const auto &val) {
          using T = std::decay_t<decltype(val)>;
          if constexpr (std::is_same_v<T, float>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT;
            out_message->value.float_val = val;
          } else if constexpr (std::is_same_v<T, int>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_INT;
            out_message->value.int_val = val;
          } else if constexpr (std::is_same_v<T, pc::types::Float3>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT3;
            out_message->value.float3_val.x = val.x;
            out_message->value.float3_val.y = val.y;
            out_message->value.float3_val.z = val.z;
          } else if constexpr (std::is_same_v<T, std::array<float, 3>>) {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_FLOAT3;
            out_message->value.float3_val.x = val[0];
            out_message->value.float3_val.y = val[1];
            out_message->value.float3_val.z = val[2];
          } else {
            out_message->value_type = POINTRECEIVER_PARAM_VALUE_UNKNOWN;
          }
        },
        param_update.value);
  } else {
    return false;
  }
  return true;
}

// Start a thread that handles networking for the point cloud
int pointrecevier_start_point_receiver(const char *pointcaster_address,
                                       int timeout_ms) {

  // initialise the zmq context if it hasn't already been initialised
  std::call_once(zmq_ctx_once,
                 [] { zmq_ctx = std::make_unique<zmq::context_t>(); });

  const std::string pointcaster_address_str(pointcaster_address);

  pointcloud_thread = std::make_unique<std::jthread>(
      [&, pointcaster_address_str, timeout_ms](std::stop_token stop_token) {
        using namespace std::chrono;
        using namespace std::chrono_literals;

        log("Beginning pointcloud receive thread");

        // create the dish that receives point clouds
        zmq::socket_t socket(*zmq_ctx, zmq::socket_type::dish);

        // TODO something in the set calls here is crashing Unity

        // don't retain frames in memory
        socket.set(zmq::sockopt::linger, 0);
        // TODO what should the high watermark be... 3 frames why?
        socket.set(zmq::sockopt::rcvhwm, 3);

        // connection attempt should time out if requested
        if (timeout_ms != 0)
          socket.set(zmq::sockopt::connect_timeout, timeout_ms);

        constexpr auto recv_timeout_ms = 100;
        socket.set(zmq::sockopt::rcvtimeo, recv_timeout_ms);

        auto endpoint = fmt::format("tcp://{}:{}", pointcaster_address_str,
                                    pointcloud_stream_port);
        socket.connect(endpoint);
        log("Listening for pointcloud stream at '{}'", endpoint);

        if (socket.handle() == nullptr) {
          log("Failed to connect");
          return;
        }

        socket.join("live");
        log("Connected");

        while (!stop_token.stop_requested()) {

          zmq::message_t incoming_msg;
          auto result = socket.recv(incoming_msg, zmq::recv_flags::none);
          if (!result) continue;

          // deserialize the incoming message
          auto msg_size = incoming_msg.size();
          bob::types::bytes buffer(msg_size);
          auto bytes_ptr = static_cast<std::byte *>(incoming_msg.data());
          buffer.assign(bytes_ptr, bytes_ptr + msg_size);
          // and place it in our queue
          cloud_queue.try_enqueue(PointCloud::deserialize(buffer));
        }

        socket.disconnect(endpoint);

        log("Disconnected");
      });

  return 0;
}

int pointreceiver_stop_point_receiver() {
  if (pointcloud_thread) {
    pointcloud_thread->request_stop();
    pointcloud_thread.reset();
  }
  return 0;
}

static PointCloud point_cloud;

bool pointreceiver_dequeue_points(int timeout_ms) {
  point_cloud = PointCloud{};
  return cloud_queue.wait_dequeue_timed(point_cloud,
                                        std::chrono::milliseconds(timeout_ms));
}

int pointreceiver_point_count() { return point_cloud.size(); }

// TODO make below compatible with C api

bob::types::color *pointColors() { return point_cloud.colors.data(); }

bob::types::position *pointPositions() { return point_cloud.positions.data(); }

std::vector<std::array<float, 4>> shader_friendly_point_positions_data;

enum class OrientationType { unchanged = 0, flip_x, flip_x_z, flip_z };
float *shaderFriendlyPointPositions(bool parallel_transform,
                                    OrientationType orientation_type) {
  std::size_t size = point_cloud.positions.size();
  shader_friendly_point_positions_data.resize(size);

  std::array<float, 3> orientation{1, 1, 1};
  if (orientation_type == OrientationType::flip_x) {
    orientation[0] = -1;
  } else if (orientation_type == OrientationType::flip_z) {
    orientation[2] = -1;
  } else if (orientation_type == OrientationType::flip_x_z) {
    orientation[0] = -1;
    orientation[2] = -1;
  }

  const auto transform_func =
      [&](const bob::types::position &pos) -> std::array<float, 4> {
    return {static_cast<float>(pos.x) * orientation[0] / 1000.0f,
            static_cast<float>(pos.y) * orientation[1] / 1000.0f,
            static_cast<float>(pos.z) * orientation[2] / 1000.0f, 0.0f};
  };

  if (parallel_transform) {
    std::transform(std::execution::par, point_cloud.positions.begin(),
                   point_cloud.positions.end(),
                   shader_friendly_point_positions_data.begin(),
                   transform_func);
  } else {
    std::transform(point_cloud.positions.begin(), point_cloud.positions.end(),
                   shader_friendly_point_positions_data.begin(),
                   transform_func);
  }

  return shader_friendly_point_positions_data.front().data();
}
}

void testMessageLoop(pointreceiver_context *ctx) {
  int count = 0;
  while (count++ < 1000) {
    pointreceiver_sync_message msg;
    // dequeue with a 5-millisecond timeout.
    if (pointreceiver_dequeue_message(ctx, &msg, 5)) {
      if (msg.message_type == POINTRECEIVER_MSG_TYPE_PARAMETER_UPDATE) {
        log("Parameter Update received: id = {}, value_type = {}", msg.id,
            (int)msg.value_type);
        switch (msg.value_type) {
        case POINTRECEIVER_PARAM_VALUE_FLOAT: log("Value: {}", msg.value.float_val); break;
        case POINTRECEIVER_PARAM_VALUE_INT: log("Value: {}", msg.value.int_val); break;
        case POINTRECEIVER_PARAM_VALUE_FLOAT3:
          log("Value: ({}, {}, {})", msg.value.float3_val.x,
              msg.value.float3_val.y, msg.value.float3_val.z);
          break;
        default: log("Unknown parameter update type"); break;
        }
      } else {
        log("Received message type: {}", (int)msg.message_type);
      }
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
  if (argc < 2) {
    // pointreceiver_start_point_receiver("127.0.0.1", 5);
    pointreceiver_start_message_receiver(ctx, "127.0.0.1");
  } else {
    // pointreceiver_start_point_receiver(argv[1], 5);
    pointreceiver_start_message_receiver(ctx, argv[1]);
  }
  // for (int i = 0; i < 3; i++) { testPointcloudLoop(); }
  for (int i = 0; i < 3; i++) { testMessageLoop(ctx); }
  pointreceiver_stop_message_receiver(ctx);
  // pointreciever_stop_point_receiver();
  pointreceiver_destroy_context(ctx);
  return 0;
}
