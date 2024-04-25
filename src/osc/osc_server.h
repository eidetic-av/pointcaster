#pragma once

#include "../logger.h"
#include "../string_utils.h"
#include "../structs.h"
#include "osc_server_config.gen.h"
#include <array>
#include <atomic>
#include <fmt/format.h>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>
#include <lo/lo.h>
#include <lo/lo_cpp.h>

namespace pc::osc {

class OscServer {
public:
  OscServer(OscServerConfiguration &config);

  void draw_imgui_window();

private:
  OscServerConfiguration &_config;
  std::unique_ptr<lo::ServerThread> _server_thread;

  void create_server(int port);
};

} // namespace pc::midi
