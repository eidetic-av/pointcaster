#include "osc_client.h"
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

OscClient::OscClient(OscClientConfiguration &config)
    : _config(config), _sender_thread([this](auto st) { send_messages(st); }) {
  // publisher::add(this);
  declare_parameters("oscclient", _config);
}

OscClient::~OscClient() {
  _sender_thread.request_stop();
  _sender_thread.join();
  // publisher::remove(this);
}

void OscClient::send_messages(std::stop_token st) {
  using namespace std::chrono_literals;

  static constexpr auto thread_wait_time = 50ms;

  // while (!_virtual_output.is_port_open() && !st.stop_requested()) {
  //   std::this_thread::sleep_for(thread_wait_time);
  // }

  // while (!st.stop_requested()) {
    //   MidiOutMessage message;
    //   while (_messages_to_publish.wait_dequeue_timed(message,
    //   thread_wait_time)) {
    //     auto status_byte =
    //         static_cast<uint8_t>(libremidi::message_type::CONTROL_CHANGE) +
    //         (message.channel_num - 1); // convert channel num to zero-indexed
    //     _virtual_output.send_message(status_byte, message.cc_num,
    //     message.value);
    //   }
    // }
}


void OscClient::draw_imgui_window() {

  ImGui::Begin("OSC Client", nullptr);

  pc::gui::draw_parameters("oscclient",
                           struct_parameters.at(std::string{"oscclient"}));

  ImGui::End();
}

} // namespace pc::osc
