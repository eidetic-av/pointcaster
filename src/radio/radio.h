#include "radio_config.h"
#include "../operators/session_operator_host.h"
#include <memory>
#include <thread>

namespace pc::radio {

class Radio {
public:
  Radio(RadioConfiguration& config, pc::operators::SessionOperatorHost& session_operator_host);

  void draw_imgui_window();

private:
  RadioConfiguration& _config;
  pc::operators::SessionOperatorHost& _session_operator_host;
  std::unique_ptr<std::jthread> _radio_thread;
};
} // namespace pc::radio
