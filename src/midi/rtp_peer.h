#pragma once

#include "../serialization.h"
#include <string>

#include <C-Binding/rtpMidi.h>

namespace pc::midi {

class RtpMidiDevice;

struct RtpPeer {
  std::string name;
  std::string address;
  uint16_t port;
  RtpMidiDevice *device;
  RTP_MIDI_PEER *peer;
  bool connected;

  RTP_MIDI_PEER *connect(
      RTP_MIDI_SESSION *session,
      std::optional<std::function<void(RTP_MIDI_PEER *, RTP_MIDI_SESSION *)>>
	  on_connect = {});

  void disconnect(
      RTP_MIDI_SESSION *session,
      std::optional<std::function<void(RTP_MIDI_PEER *, RTP_MIDI_SESSION *)>>
          on_disconnect = {});
};

struct RtpPeerKey {
  std::string name;
  std::string address;
  uint16_t port;

  bool operator<(const RtpPeerKey &other) const {
    if (name < other.name) return true;
    if (name > other.name) return false;
    if (address < other.address) return true;
    if (address > other.address) return false;
    return port < other.port;
  }

  bool operator==(const RtpPeerKey &other) const {
    return name == other.name && address == other.address && port == other.port;
  }

  DERIVE_SERDE(RtpPeerKey, (&Self::name, "name")
	       (&Self::address, "address")
	       (&Self::port, "port"))
};

} // namespace pc::midi
