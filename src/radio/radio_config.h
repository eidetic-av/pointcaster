#pragma once

#include "../serialization.h"
// #include "../structs.h"

namespace pc::radio {

struct RadioConfiguration {
  int port = 9999;
  bool enabled = false;
  bool compress_frames;
  bool capture_stats;

  bool operator==(const RadioConfiguration other) const {
    return port == other.port && compress_frames == other.compress_frames &&
	   capture_stats == other.capture_stats;
  }
  bool operator!=(const RadioConfiguration other) const {
    return !operator==(other);
  }

  DERIVE_SERDE(RadioConfiguration,
	       (&Self::port, "port")
	       (&Self::enabled, "enabled")
	       (&Self::compress_frames, "compress_frames")
	       (&Self::capture_stats, "capture_stats"))

  using MemberTypes = pc::reflect::type_list<int, bool, bool, bool>;
  static const std::size_t MemberCount = 4;
};

} // namespace pc::radio
