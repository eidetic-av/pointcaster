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

using namespace pc::parameters;

OscServer::OscServer(OscServerConfiguration &config) : _config(config) {
  _config.port = std::clamp(_config.port, 1024, 49151);
  if (_config.enable) {
    create_server(_config.port);
  }
  declare_parameters("oscserver", _config);
}

template <typename T> void receive_msg(std::string_view address, T value) {
  if constexpr (std::is_same_v<T, std::string_view>) {
    pc::logger->debug("{}: s {}", address, value);
    return;
  }
  std::string topic{address.substr(1, -1)};
  std::replace(topic.begin(), topic.end(), '/', '.');

  if constexpr (types::VectorType<T>) {
    // if the osc message is setting the entire vector
    if (parameter_bindings.contains(std::string_view{topic})) {
      set_parameter_value(topic, value);
    }
  } else {
    // if the osc message is setting individual elements of a vector
    std::optional<int> vector_element;
    if (strings::ends_with(topic, ".0") || strings::ends_with(topic, ".x")) {
      vector_element = 0;
    } else if (strings::ends_with(topic, ".1") ||
	       strings::ends_with(topic, ".y")) {
      vector_element = 1;
    } else if (strings::ends_with(topic, ".2") ||
	       strings::ends_with(topic, ".z")) {
      vector_element = 2;
    } else if (strings::ends_with(topic, ".3") ||
	       strings::ends_with(topic, ".w")) {
      vector_element = 3;
    }
    if (!vector_element) return;
    topic = topic.substr(0, topic.length() - 2);
    std::visit(
	[&](auto &&parameter_value_ref) {
	  auto &parameter = parameter_value_ref.get();
	  using ParamType = std::decay_t<decltype(parameter)>;
	  if constexpr (types::VectorType<ParamType>) {
	    if constexpr (std::is_same_v<typename ParamType::vector_type, T>) {
              // if the parameter we are trying to set is a vector
              // and we are setting it with the element type of the vector,
	      // then we can get the vector and set just the element
	      parameter[*vector_element] = value;
	    }
          }
	},
	parameter_bindings.at(topic).value);
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
	  receive_msg(path, argv[0]->f);
        }
        if (std::strcmp(types, "fff") == 0) {
	  receive_msg(path, Float3{argv[0]->f, argv[1]->f, argv[2]->f});
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

  auto enabled = _config.enable;

  if (pc::gui::draw_parameters(
	  "oscserver", struct_parameters.at(std::string{"oscserver"}))) {

    _config.port = std::clamp(_config.port, 1024, 49151);

    if (enabled && !_config.enable) {
      _server_thread->stop();
    } else if (!enabled && _config.enable) {
      _server_thread->start();
    }

  }

  ImGui::End();
}

} // namespace pc::osc
