#include "pointreceiver.h"
#include <chrono>
#include <iostream>
#include <mutex>
#include <numeric>
#include <readerwriterqueue/readerwritercircularbuffer.h>
#include <thread>
#include <vector>

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
  dish_thread =
      std::make_unique<std::thread>([&, point_caster_address, timeout_ms]() {
        using namespace zmq;
        using namespace std::chrono;
        using namespace std::chrono_literals;

        log("Beginning networking thread");
        // create the dish that receives point clouds
        context_t ctx;
        socket_t dish(ctx, socket_type::dish);
        // don't retain frames in memory
        dish.set(sockopt::linger, 0);
        dish.set(sockopt::rcvhwm, 10);
        // connection attempt should time out if requested
        // if (timeout_ms != 0)
        //  dish.set(sockopt::connect_timeout, timeout_ms);
        dish.set(sockopt::connect_timeout, 0);

        // auto endpoint = fmt::format("tcp://{}", point_caster_address);
        auto endpoint = "tcp://127.0.0.1:9999";
        log(fmt::format("Attempting to connect to '{}'", endpoint));
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
          // block until we receive a pointcloud message
          zmq::message_t incoming_msg;
          auto result = dish.recv(incoming_msg);
          std::string group = incoming_msg.group();
          // deserialize the incoming message
          auto msg_size = incoming_msg.size();
          bob::types::bytes buffer(msg_size);
          auto bytes_ptr = static_cast<std::byte *>(incoming_msg.data());
          buffer.assign(bytes_ptr, bytes_ptr + msg_size);
          // log(fmt::format("bytes size: {}", buffer.size()));
          // log(fmt::format("pc.size: {}", point_cloud.size()));
          if (group == "live")
            cloud_queue.try_enqueue(PointCloud::deserialize(buffer));
          else if (group == "snapshots") {
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
  dish_thread.reset(nullptr);
  return 0;
}

static PointCloud point_cloud;

bool dequeue() {
  point_cloud = PointCloud{};
  const bool dequeue_success = cloud_queue.try_dequeue(point_cloud);
  if (!dequeue_success)
    return false;
  point_cloud += synthesized_snapshot_frames;
  return true;
}

int pointCount() { return point_cloud.size(); }
bob::types::position *pointPositions() { return point_cloud.positions.data(); }
bob::types::color *pointColors() { return point_cloud.colors.data(); }
}

void testLoop() {
  startNetworkThread();
  int i = 0;
  while (i++ < 6000) {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(0.5ms);
    if (!dequeue())
      continue;
    auto count = pointCount();
    // auto buffer = pointPositions();
    log(fmt::format("--{}", i));
    for (auto frame : snapshot_frames) {
      log("f: " + std::to_string(frame.positions[0].x));
    }
    // auto o = buffer[200];
    // log(fmt::format("x {}, y {}, z {}, p {}", o.x, o.y, o.z, o.__pad));
  }
  stopNetworkThread();
}

int main() {
  for (int i = 0; i < 3; i++) {
    testLoop();
  }
  return 0;
}
