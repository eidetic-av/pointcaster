#pragma once

#include <rfl/DefaultVal.hpp>
#include <rfl/Literal.hpp>

namespace pc::publishers {

struct MessageStreamerConfiguration {
  rfl::DefaultVal<int> port = 9991; // @minmax(1024, 49151)

  using Tag = rfl::Literal<"message_streamer">;
};

} // namespace pc::publishers