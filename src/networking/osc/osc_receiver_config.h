#pragma once

#include <rfl/DefaultVal.hpp>

namespace pc::networking::osc {

struct OscReceiverConfiguration {
  rfl::DefaultVal<bool> enable = true;
  rfl::DefaultVal<int>  port   = 9001;
};

} // namespace pc::networking::osc
