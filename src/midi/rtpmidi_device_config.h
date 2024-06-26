#pragma once

#include "rtp_peer.h"
#include <vector>

namespace pc::midi {

using peer_set = std::vector<RtpPeerKey>;

struct RtpMidiDeviceConfiguration {
  bool enable = true;
  std::string ip = "0.0.0.0";
  peer_set peers; // @optional
};

} // namespace pc::midi
