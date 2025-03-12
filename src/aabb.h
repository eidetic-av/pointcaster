#pragma once
#include "structs.h"

namespace pc {
struct AABB {
  pc::types::Float3 min{};
  pc::types::Float3 max{};

  AABB() = default;
  AABB(const pc::types::Float3 &min, const pc::types::Float3 &max)
      : min(min), max(max) {}

  // Function to get the center of the AABB
  pc::types::Float3 center() const { return (min + max) / 2.0f; }

  // Function to get the extents of the AABB
  pc::types::Float3 extents() const { return (max - min) / 2.0f; }

  pc::types::MinMax<pc::types::Float3> toMinMax() const { return {min, max}; }
};
} // namespace pc