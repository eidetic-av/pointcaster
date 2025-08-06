#pragma once

#include "../device_config.gen.h"

namespace pc::devices {

struct PlySequencePlayer;

struct PlyStreamingConfiguration {
  bool use_static_channel = false; // @optional
  bool requires_update = true; // @hidden
  bool disable_compression = false; // @optional
};

struct PlySequencePlayerConfiguration {
  std::string id;
  bool active = true;
  std::string directory; // @disabled
  bool playing = true; // @hidden
  int frame_rate = 30;
  int current_frame = 0;
  int buffer_capacity = 450; // @minmax(1, 1800)
  int io_threads = 4; // @minmax(1, 12)
  bool looping = true;
  int start_frame = 0; // @minmax(0, 20000)

  DeviceTransformConfiguration transform; // @optional
  ColorConfiguration color; // @optional
  PlyStreamingConfiguration streaming; // @optional

  using DeviceType = PlySequencePlayer;
  static constexpr auto PublishPath = "ply";
};

} // namespace pc::devices