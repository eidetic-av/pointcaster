#include "rtp_peer.h"
#include "../logger.h"

#ifdef WIN32
#include <Winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#endif

namespace pc::midi {

RTP_MIDI_PEER* RtpPeer::connect(
    RTP_MIDI_SESSION *session,
    std::optional<std::function<void(RTP_MIDI_PEER *, RTP_MIDI_SESSION *)>>
	on_connect) {
  in_addr peer_address;
  if (inet_pton(AF_INET, address.data(), &peer_address) <= 0) {
    throw std::runtime_error("Failed to convert peer address");
  }
  peer = rtpMidiSessionPeerAdd(session, &peer_address, sizeof(peer_address),
                               port, 0, this);
  if (on_connect.has_value()) {
    (*on_connect)(peer, session);
  }
  connected = true;
  return peer;
}

void RtpPeer::disconnect(
    RTP_MIDI_SESSION *session,
    std::optional<std::function<void(RTP_MIDI_PEER *, RTP_MIDI_SESSION *)>>
	on_disconnect) {
  connected = false;

  if (on_disconnect.has_value()) {
    (*on_disconnect)(peer, session);
  }
  rtpMidiSessionPeerRemove(session, peer);
}

} // namespace pc::midi
