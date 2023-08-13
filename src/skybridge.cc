#include "skybridge.h"
#include <spdlog/spdlog.h>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXUserAgent.h>
#include <thread>

namespace pc::skybridge {

	using namespace ix;

	std::string server_endpoint = "ws://localhost:8080/";
	std::atomic<bool> run_status_thread = true;

	void initConnection() {
		std::thread skybridge_status_thread([&]() {
			pc::logger->info("Initialising Skybridge connection...");
			initNetSystem();
			ix::WebSocket socket;
			pc::logger->info("Connecting to server at '{}'", server_endpoint);
			socket.setUrl(server_endpoint);
			// send keep-alive message every 30s
			socket.setPingInterval(30);
			socket.start();
			while (run_status_thread) {
			    using namespace std::chrono_literals;
				std::this_thread::sleep_for(500ms);
			}
			socket.stop();
			uninitNetSystem();
		});
		skybridge_status_thread.detach();
	}

}
