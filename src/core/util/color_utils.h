#pragma once

#include <algorithm>
#include <cmath>
#include <pointcaster/core_types.h>

namespace pc {

// places a normalised value on a fully saturated hue circle
inline color hue_color(float hue) {
  const float sector = std::clamp(hue, 0.0f, 1.0f) * 6.0f;
  const float fraction = sector - std::floor(sector);
  const auto channel = [](float value) {
    return static_cast<unsigned char>(std::lround(value * 255.0f));
  };
  switch (static_cast<int>(sector)) {
  case 0:
    return {255, channel(fraction), 0, 255};
  case 1:
    return {channel(1.0f - fraction), 255, 0, 255};
  case 2:
    return {0, 255, channel(fraction), 255};
  case 3:
    return {0, channel(1.0f - fraction), 255, 255};
  case 4:
    return {channel(fraction), 0, 255, 255};
  default:
    return {255, 0, channel(1.0f - fraction), 255};
  }
}

} // namespace pc
