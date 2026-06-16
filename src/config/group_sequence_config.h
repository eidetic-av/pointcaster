#pragma once

#include <rfl/DefaultVal.hpp>

namespace pc {
struct GroupSequenceConfiguration {
  rfl::DefaultVal<bool> playing = true;   // @hidden
  rfl::DefaultVal<int> current_frame = 0; // @hidden
};
} // namespace pc
