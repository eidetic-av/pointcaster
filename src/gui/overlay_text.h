#pragma once

#include <string>
#include "../structs.h"

namespace pc::gui {

using pc::types::Int2;

struct OverlayText {
  std::string text;
  Int2 position;
  float scale = 1.0f;
};

} // namespace pc::gui
