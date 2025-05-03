#pragma once

#include "../device_config.gen.h"

namespace pc::devices {

using pc::devices::DeviceTransformConfiguration;

struct PlySequencePlayer;

struct PlySequencePlayerConfiguration {
  std::string id;
  bool active = true;
  std::string directory; // @disabled
  bool playing = true; // @hidden
  int frame_rate = 30;
  int current_frame = 0;

  DeviceTransformConfiguration transform; // @optional

  using DeviceType = PlySequencePlayer;
  static constexpr auto PublishPath = "ply";
};

} // namespace pc::devices