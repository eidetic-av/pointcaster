#pragma once

#include <pointcaster/core_types.h>
#include <rfl/DefaultVal.hpp>

namespace pc {

struct ColorTransformConfiguration {
  rfl::DefaultVal<float> gain = 1; // @minmax(0, 15)
};

} // namespace pc