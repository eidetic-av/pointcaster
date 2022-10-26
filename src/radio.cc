#include "radio.h"
#include <chrono>
#include <map>
#define ZMQ_BUILD_DRAFT_API
#include <zmq.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include "./devices/device.h"
// #include <yaclib/executor/thread_pool.hpp>
// #include <yaclib/executor/submit.hpp>

namespace bob::pointcaster {

Radio::Radio(const int _port) : port(_port) {

  radio_thread = std::jthread([&](std::stop_token st) {
    using namespace std::chrono_literals;

    constexpr auto broadcast_rate = 16ms;

    zmq::context_t zmq_context;
    zmq::socket_t radio(zmq_context, zmq::socket_type::radio);
    // prioritise the latest frame
    radio.set(zmq::sockopt::sndhwm, 1);
    // and don't keep excess frames in memory
    radio.set(zmq::sockopt::linger, 0);

    auto destination = fmt::format("tcp://*:{}", port);
    // auto destination = fmt::format("tcp://192.168.1.10:{}", port);
    radio.bind(destination);
    spdlog::info("Radio broadcasting on port {}", port);

    constexpr auto worker_thread_count = 5;

    spdlog::info("Initialising Radio worker thread pool ({} threads)",
		 worker_thread_count);
    // auto worker_threads = yaclib::MakeThreadPool(worker_thread_count);

    while (!st.stop_requested()) {

      std::this_thread::sleep_for(broadcast_rate);
      
      // tell one thread to wait the length of our broadcast rate

      // Submit(*worker_threads,
      // 	     [&] { std::this_thread::sleep_for(broadcast_rate); });

      // tell one thread to get, serialize and send our synthesized point
      // cloud

      // Submit(*worker_threads, [&] {
	auto point_cloud = bob::sensors::synthesizedPointCloud();
	if (point_cloud.size() == 0) continue;
	constexpr auto compress = false;
	auto bytes = point_cloud.serialize(compress);
	zmq::message_t point_cloud_msg(bytes);
	point_cloud_msg.set_group("a");
	radio.send(point_cloud_msg, zmq::send_flags::none);
      // });

      // for each individual device, broadcast cams
      // for (auto& attached_device : bob::sensors::attached_devices) {
      //   Submit(*worker_threads, [&] {
      //     auto point_cloud = attached_device->getPointCloud();
      //   });
      // }
      // block until all worker threads have completed
      // before continuing our loop

      // worker_threads->Stop();
      // spdlog::info("Waiting...");
      // worker_threads->Wait();
      // spdlog::info("End");
    }

    spdlog::info("Ending broadcast");
  });
}

  Radio::~Radio() {
    radio_thread.request_stop();
  }

}
