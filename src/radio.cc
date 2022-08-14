#include "radio.h"
#include <chrono>
#include <map>
#define ZMQ_BUILD_DRAFT_API
#include <zmq.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include "./devices/device.h"
#include <yaclib/executor/thread_pool.hpp>
#include <yaclib/executor/submit.hpp>

namespace bob::pointcaster {

Radio::Radio(const int _port) : port(_port) {

  radio_thread = std::jthread([&](std::stop_token st) {
    using namespace zmq;
    using namespace std::chrono_literals;

    constexpr auto broadcast_rate = 16ms;

    context_t zmq_context;
    socket_t radio(zmq_context, socket_type::radio);
    // prioritise the latest frame
    radio.set(sockopt::sndhwm, 1);
    // and don't keep excess frames in memory
    radio.set(sockopt::linger, 0);

    auto destination = fmt::format("tcp://*:{}", port);
    radio.bind(destination);
    spdlog::info("Radio broadcasting on port {}", port);

    constexpr auto worker_thread_count = 5;
    spdlog::info("Initialising Radio worker thread pool ({} threads)",
		 worker_thread_count);
    auto worker_threads = yaclib::MakeThreadPool(worker_thread_count);

    while (!st.stop_requested()) {
      // tell one thread to wait the length of our broadcast rate
      Submit(*worker_threads,
	     [&] { std::this_thread::sleep_for(broadcast_rate); });
      // tell one thread to get, serialize and send our synthesized point
      // cloud
      Submit(*worker_threads, [&] {
	auto point_cloud = bob::sensors::synthesizedPointCloud();
	constexpr auto compress = false;
	message_t point_cloud_msg(point_cloud.serialize(compress));
	point_cloud_msg.set_group("a");
	radio.send(point_cloud_msg, send_flags::none);
      });
      // for each individual device, broadcast cams
      // for (auto& attached_device : bob::sensors::attached_devices) {
      //   Submit(*worker_threads, [&] {
      //     auto point_cloud = attached_device->getPointCloud();
      //   });
      // }
      // block until all worker threads have completed
      // before continuing our loop
      worker_threads->Stop();
      worker_threads->Wait();
    }
  });
}

}
