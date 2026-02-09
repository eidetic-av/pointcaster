#include "osc_client.h"
#include "../gui/widgets.h"
#include "../logger/logger.h"
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

OscClient::OscClient(OscClientConfiguration &config)
    : _config(config), _sender_thread([this](auto st) { send_messages(st); }) {
  publisher::add(this);
  parameters::declare_parameters("workspace", "oscclient", _config);
}

OscClient::~OscClient() {
  _sender_thread.request_stop();
  _sender_thread.join();
  publisher::remove(this);
}

void OscClient::send_messages(std::stop_token st) {
  using namespace std::chrono_literals;

  static constexpr auto thread_wait_time = 50ms;
  lo::Address osc_client(_config.address, _config.port);

  while (!st.stop_requested()) {
    try {
      OscMessage msg;
      while (_messages_to_publish.wait_dequeue_timed(msg, thread_wait_time)) {
        auto &address = msg.address;
        if (address.empty()) {
          pc::logger()->error("empty OSC message address");
          return;
        }
        // ensure that the osc address starts with a leading forward slash
        if (address.front() != '/') { address.insert(address.begin(), '/'); }
        osc_client.send(address, msg.value);
      }
    } catch (const std::exception &e) {
      pc::logger()->error("OSC client send failed: {}", e.what());
    } catch (...) {
      pc::logger()->error("OSC client send failed with unknown exception");
    }
  }
}

void OscClient::draw_imgui_window() {

  ImGui::Begin("OSC Client", nullptr);

  pc::gui::draw_parameters(
      "oscclient", parameters::struct_parameters.at(std::string{"oscclient"}));

  ImGui::End();
}

} // namespace pc::osc
