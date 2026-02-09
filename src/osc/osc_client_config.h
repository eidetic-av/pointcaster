#pragma once

#include <fmt/format.h>
#include <map>
#include <string>
#include <tuple>

namespace pc::osc {

struct OscClientConfiguration {
  bool show_window = false;
  bool enable = true;
  std::string address = "127.0.0.1";
  int port = 9000;
};

} // namespace pc::osc
