#pragma once

#include <rfl/DefaultVal.hpp>

namespace pc::networking {

struct PointStreamerConfiguration {
  rfl::DefaultVal<int> publish_hz = 60;
  rfl::DefaultVal<std::string> address = "*";
  rfl::DefaultVal<int> port = 9992;
  rfl::DefaultVal<bool> compress = false;
};

} // namespace pc::networking