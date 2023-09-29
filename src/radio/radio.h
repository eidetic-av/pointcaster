#include <memory>
#include <thread>
#include "radio_config.h"

namespace pc::radio {

class Radio {
public:
  Radio(RadioConfiguration& config);

  void draw_imgui_window();

private:
  RadioConfiguration& _config;
  std::unique_ptr<std::jthread> _radio_thread;
};
} // namespace pc::radio
