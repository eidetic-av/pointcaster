#pragma once

#include <rfl/DefaultVal.hpp>

namespace pc::receivers {

struct OscReceiverConfiguration {
  rfl::DefaultVal<bool> enable = true;
  rfl::DefaultVal<int> port = 9001;
};

} // namespace pc::receivers
