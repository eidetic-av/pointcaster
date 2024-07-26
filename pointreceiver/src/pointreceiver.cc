#include "pointreceiver.h"
#include <chrono>
#include <iostream>
#include <mutex>
#include <numeric>
#include <readerwriterqueue/readerwritercircularbuffer.h>
#include <thread>
#include <vector>
#include <execution>
#define ZMQ_BUILD_DRAFT_API
#include <zmq.hpp>

extern "C" {

std::atomic<bool> request_thread_stop = false;
std::unique_ptr<std::thread> dish_thread;

std::mutex buffer_access;

using bob::types::PointCloud;
static moodycamel::BlockingReaderWriterCircularBuffer<PointCloud>
    cloud_queue(1);
static std::vector<PointCloud> snapshot_frames;
static PointCloud synthesized_snapshot_frames;

// Start a thread that handles networking
int startNetworkThread(const char *point_caster_address, int timeout_ms) {
  request_thread_stop = false;
  const std::string point_caster_address_str(point_caster_address);
  dish_thread =
      std::make_unique<std::thread>([&, point_caster_address_str, timeout_ms]() {
        using namespace std::chrono;
        using namespace std::chrono_literals;

        log("Beginning networking thread");

        // create the dish that receives point clouds
        zmq::context_t ctx;
        zmq::socket_t dish(ctx, zmq::socket_type::dish);

	// TODO something in the set calls here is crashing Unity

        // don't retain frames in memory
        dish.set(zmq::sockopt::linger, 0);
        // dish.set(zmq::sockopt::rcvhwm, 3);

        // connection attempt should time out if requested
        if (timeout_ms != 0) dish.set(zmq::sockopt::connect_timeout, timeout_ms);

	    constexpr auto recv_timeout_ms = 100;
	    dish.set(zmq::sockopt::rcvtimeo, recv_timeout_ms);

        auto endpoint = fmt::format("tcp://{}", point_caster_address_str);
        fmt::print("Attempting to connect to '{}'", endpoint);
        dish.connect(endpoint);

        if (dish.handle() == nullptr) {
          log("Failed to connect");
          return;
        }

        log("Connected");
        dish.join("live");
        dish.join("snapshots");
        log("Joined live and snapshots groups");

        time_point<system_clock> last_snapshot_time = system_clock::now();

        while (!request_thread_stop) {

          zmq::message_t incoming_msg;
          auto result = dish.recv(incoming_msg, zmq::recv_flags::none);
	      if (!result) continue;

          std::string group = incoming_msg.group();

          // deserialize the incoming message
          auto msg_size = incoming_msg.size();
          bob::types::bytes buffer(msg_size);
          auto bytes_ptr = static_cast<std::byte *>(incoming_msg.data());
          buffer.assign(bytes_ptr, bytes_ptr + msg_size);
          // log(fmt::format("bytes size: {}", buffer.size()));
          // log(fmt::format("pc.size: {}", point_cloud.size()));
          if (group == "live") {
            cloud_queue.try_enqueue(PointCloud::deserialize(buffer));
	  } else if (group == "snapshots") {
	    auto msg_begin = static_cast<const char *>(incoming_msg.data());
            std::string header_msg(msg_begin, msg_begin + 5);
            if (header_msg == "clear") {
              snapshot_frames.clear();
              synthesized_snapshot_frames = PointCloud{};
            } else {
              auto frame_time = system_clock::now();
              if (frame_time - last_snapshot_time > 1000ms) {
                // snapshot_frames.push_back(PointCloud::deserialize(buffer));
                // synthesized_snapshot_frames =
                // std::reduce(snapshot_frames.begin(), snapshot_frames.end(),
                // PointCloud{}, [](auto a, auto b) -> PointCloud { return a +
                // b; });
                last_snapshot_time = frame_time;
                synthesized_snapshot_frames = PointCloud::deserialize(buffer);
              }
            }
          }
        }

        dish.disconnect(endpoint);

        // cleanup any snapshots
        snapshot_frames.clear();
        synthesized_snapshot_frames = PointCloud{};

        log("Disconnected");
      });

  return 0;
}

int stopNetworkThread() {
  request_thread_stop = true;
  dish_thread->join();
  return 0;
}

static PointCloud point_cloud;

bool dequeue(int timeout_ms) {
  point_cloud = PointCloud{};
  return cloud_queue.wait_dequeue_timed(point_cloud, std::chrono::milliseconds(timeout_ms));
}

int pointCount() { return point_cloud.size(); }
bob::types::color *pointColors() { return point_cloud.colors.data(); }

bob::types::position *pointPositions() { return point_cloud.positions.data(); }

std::vector<std::array<float, 4>> shader_friendly_point_positions_data;

float* shaderFriendlyPointPositions(bool parallel_transform, OrientationType orientation_type) {
    std::size_t size = point_cloud.positions.size();
    shader_friendly_point_positions_data.resize(size);

    std::array<float, 3> orientation{ 1, 1, 1 };
    if (orientation_type == OrientationType::flip_x) 
    {
        orientation[0] = -1;
    }
    else if (orientation_type == OrientationType::flip_x_z)
    {
        orientation[0] = -1;
        orientation[2] = -1;
    }

	const auto transform_func = [&](const bob::types::position& pos)->std::array<float, 4> {
		return { static_cast<float>(pos.x) * orientation[0] / 1000.0f,
			     static_cast<float>(pos.y) * orientation[1] / 1000.0f,
			     static_cast<float>(pos.z) * orientation[2] / 1000.0f, 0.0f };
	};

    if (parallel_transform) {
        std::transform(std::execution::par, point_cloud.positions.begin(), point_cloud.positions.end(), 
                       shader_friendly_point_positions_data.begin(), transform_func);
    } else {
        std::transform(point_cloud.positions.begin(), point_cloud.positions.end(), 
                       shader_friendly_point_positions_data.begin(), transform_func);
    }

    return shader_friendly_point_positions_data.front().data();
}

}

void testLoop() {
  int i = 0;
  while (i++ < 6000) {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(0.5ms);
    if (!dequeue()) continue;
    auto count = pointCount();
    // auto buffer = pointPositions();
    log(fmt::format("--{}", i));
    for (auto frame : snapshot_frames) {
      log("f: " + std::to_string(frame.positions[0].x));
    }
    // auto o = buffer[200];
    // log(fmt::format("x {}, y {}, z {}, p {}", o.x, o.y, o.z, o.__pad));
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) startNetworkThread();
  else startNetworkThread(argv[1]);
  for (int i = 0; i < 3; i++) {
    testLoop();
  }
  stopNetworkThread();
  return 0;
}
