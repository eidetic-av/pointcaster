#pragma once
#include <rfl/DefaultVal.hpp>

namespace pc {

struct SequenceConfiguration {
  rfl::DefaultVal<bool> playing = true; // @hidden
  rfl::DefaultVal<bool> looping = true;
  rfl::DefaultVal<int> frame_rate = 30;
  rfl::DefaultVal<int> start_frame = 0;      // @minmax(0, 999999999)
  rfl::DefaultVal<int> current_frame = 0;      // @minmax(0, 999999999)
  rfl::DefaultVal<int> end_frame = -1;       // @minmax(-1, 999999999)
  rfl::DefaultVal<int> buffer_capacity = 60; // @minmax(8, 600)
  rfl::DefaultVal<int> prefetch_ahead = 30;  // @minmax(1, 300)
};

} // namespace pc