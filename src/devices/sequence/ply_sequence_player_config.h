#pragma once

#include "../device_config.gen.h"

namespace pc::devices {

using pc::devices::DeviceTransformConfiguration;

struct PlySequencePlayer;

struct PlySequencePlayerConfiguration {
  using DeviceType = PlySequencePlayer;
  std::string id;
  bool active = true;
  std::string directory; // @disabled
  bool playing = true; // @hidden
  int frame_rate = 30;
  int current_frame = 0;

  DeviceTransformConfiguration transform; // @optional
};

} // namespace pc::devices