#pragma once

#include <fmt/format.h>
#include <map>
#include <string>
#include <tuple>

namespace pc::osc {

struct OscServerConfiguration {
  bool show_window = false;
  bool enable = true;
  int port = 9001;
};

} // namespace pc::osc
