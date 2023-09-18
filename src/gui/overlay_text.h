#pragma once

#include <string>
#include "../structs.h"

namespace pc::gui {

using pc::types::int2;

struct OverlayText {
  std::string text;
  int2 position;
  float scale = 1.0f;
};

} // namespace pc::gui
