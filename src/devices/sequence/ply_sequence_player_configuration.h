#pragma once

#include "../device_config.gen.h"

namespace pc::devices {

using uid = pc::types::uid;
using pc::types::Float3;

struct PlySequencePlayerConfiguration {
  std::string id;
  bool active = true;
  std::string directory;
  bool playing = true;
  int current_frame = 0;
  Float3 translate; // @minmax(-5000, 5000)
};

} // namespace pc::devices