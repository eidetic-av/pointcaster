#pragma once

#include "../device_config.gen.h"

namespace pc::devices {

using uid = pc::types::uid;
using pc::types::Float3;

struct PlySequencePlayerConfiguration {
  std::string id;
  std::string directory;
  Float3 translate; // @minmax(-5000, 5000)
};

} // namespace pc::devices