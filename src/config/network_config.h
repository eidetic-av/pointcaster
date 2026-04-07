#pragma once

#include <rfl/DefaultVal.hpp>
#include <rfl/Skip.hpp>
#include <string>

namespace pc {

struct NetworkConfiguration {
  rfl::DefaultVal<std::string> ip_address = "";
  rfl::DefaultVal<std::string> subnet_mask = "255.255.255.0";
  rfl::DefaultVal<std::string> gateway_address = "192.168.1.1";
  rfl::Skip<bool> apply;
};

} // namespace pc
