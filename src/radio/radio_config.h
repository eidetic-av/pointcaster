#pragma once

#include "../serialization.h"
#include "../structs.h"

namespace pc::radio {

struct RadioConfiguration {
  std::string address = "*";
  int port = 9992;
  int static_channel_port = 9993;
  bool enabled = false;
  bool compress_frames;
  bool capture_stats;
};

  // bool operator==(const RadioConfiguration other) const {
  //   return port == other.port && compress_frames == other.compress_frames &&
  // 	   capture_stats == other.capture_stats;
  // }
  // bool operator!=(const RadioConfiguration other) const {
  //   return !operator==(other);
  // }

} // namespace pc::radio
