#include "skybridge.h"
#include <spdlog/spdlog.h>
#include <zmq.hpp>
#include <thread>
#include <string>

namespace bob::skybridge {

	using namespace zmq;

	std::string server_endpoint = "tcp://127.0.0.1:3000";

	void initSubscriber() {
		std::thread skybridge_subscriber_thread([]() {
			context_t zmq_context;
			// create the subscriber
			socket_t subscriber(zmq_context, socket_type::xsub);
			subscriber.connect(server_endpoint);
			subscriber.set(zmq::sockopt::subscribe, "");
			spdlog::info("Connected to Skybridge server");

			// TODO better thread quit handle
			while (true) {
				zmq::message_t incoming_msg;
				auto result = subscriber.recv(incoming_msg);
				/*auto raw_msg = reinterpret_cast<char*>(incoming_msg.data());
				spdlog::info(raw_msg);*/
			}
		});
		skybridge_subscriber_thread.detach();
	}

}