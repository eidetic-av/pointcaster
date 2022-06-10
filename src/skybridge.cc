#include "skybridge.h"
#include <spdlog/spdlog.h>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXUserAgent.h>

namespace bob::skybridge {

	using namespace ix;

	std::string server_endpoint = "ws://127.0.0.1:8080";

	void initSubscriber() {
#if (_WIN32)
		initNetSystem();
#endif

	}

}