#include "osc_server.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../parameters.h"
#include "../publisher/publisher.h"
#include "../string_utils.h"
#include <algorithm>
#include <cmath>
#include <imgui.h>
#include <map>
#include <memory>
#include <vector>

namespace pc::osc {

OscServer::OscServer(OscServerConfiguration &config) : _config(config) {
  _config.port = std::clamp(_config.port, 1024, 49151);
  create_server(_config.port);
  declare_parameters("oscserver", _config);
}

template <typename T> void receive_msg(std::string_view address, T value) {
  if constexpr (std::is_same_v<T, std::string_view>) {
    pc::logger->debug("{}: s {}", address, value);
  } else {
    std::string topic{address.substr(1, -1)};
    std::replace(topic.begin(), topic.end(), '/', '.');
    // pc::logger->debug("trying {} to {}", topic, value);
    if (parameter_bindings.contains(std::string_view{topic})) {
      // pc::logger->debug("setting {} to {}", topic, value);
      set_parameter_value(topic, value);
    }
  }
}

void OscServer::create_server(int port) {
  _server_thread = std::make_unique<lo::ServerThread>(_config.port);
  // adding a nullptr, nullpt method means it just matches on all messages
  _server_thread->add_method(
      nullptr, nullptr,
      [](const char *path, const char *types, lo_arg **argv, int argc) {
        // if (argc > 1) {
	if (std::strcmp(types, "f") == 0) {
	  if (argc == 3) {
	    receive_msg(path, Float3{argv[0]->f, argv[1]->f, argv[2]->f});
	  }
	}
        // } else {
        //   if (std::strcmp(types, "i") == 0) {
        //     receive_msg(path, argv[0]->i);
        //   }
        //   else if (std::strcmp(types, "f") == 0) {
        //     receive_msg(path, argv[0]->f);
        //   }
        //   else if (std::strcmp(types, "s") == 0) {
        //     receive_msg(path, std::string_view(&argv[0]->s));
        //   }
        // }
      });
  _server_thread->start();
  pc::logger->info("OSC Server listening on port {}", port);
}

void OscServer::draw_imgui_window() {

  ImGui::Begin("OSC Server", nullptr);

  if (pc::gui::draw_parameters(
	  "oscserver", struct_parameters.at(std::string{"oscserver"}))) {

    _config.port = std::clamp(_config.port, 1024, 49151);
    // 

  }

  ImGui::End();
}

} // namespace pc::osc
