#pragma once

#include <rfl/DefaultVal.hpp>
#include <string>
#include <vector>

namespace pc::networking {

struct StreamChannelOverride {
  std::string address;
  rfl::DefaultVal<bool> enabled = true;
};

struct PointStreamerConfiguration {
  rfl::DefaultVal<int> publish_hz = 60;
  rfl::DefaultVal<std::string> address = "*";
  rfl::DefaultVal<int> port = 9992;
  rfl::DefaultVal<bool> compress = false;
  std::vector<StreamChannelOverride> channels{}; // @hidden
};

} // namespace pc::networking