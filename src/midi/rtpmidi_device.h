#pragma once

#include "../serialization.h"
#include "../logger.h"
#include "rtp_peer.h"
#include <chrono>
#include <map>
#include <netinet/in.h>
#include <thread>
#include <netinet/in.h>
#include "rtpmidi_device_config.gen.h"

#include "/opt/libRtpMidiSDK/C-Binding/rtpMidi.h"

namespace pc::midi {
class RtpMidiDevice {

public:

  static constexpr uint16_t rtp_midi_ctrl_port = 5004;
  static constexpr uint16_t rtp_midi_data_port = 5005;
  static constexpr auto rtp_tick_time_ms = 100;

  RtpMidiDevice(RtpMidiDeviceConfiguration &config);
  ~RtpMidiDevice();

  RtpMidiDevice(const RtpMidiDevice &&) = delete;
  RtpMidiDevice &operator=(const RtpMidiDevice &&) = delete;

  RtpMidiDevice(RtpMidiDevice &&) = delete;
  RtpMidiDevice &operator=(RtpMidiDevice &&) = delete;

  template <typename... Bytes>
  void send_message(Bytes... bytes) {
    // create a byte array from the variadic arguments
    std::array<uint8_t, sizeof...(bytes)> arr{static_cast<uint8_t>(bytes)...};
    rtpMidiSessionSendMidiData(_rtp_midi_session, 0, arr.data(), arr.size());
  }

  void draw_imgui();

  int ctrl_socket() { return _ctrl_socket; }
  int data_socket() { return _data_socket; }

  RtpPeer *get_or_create_peer(std::string_view name, std::string_view address,
			      uint16_t port);

  bool mdns_stop_requested() {
    return _mdns_thread.get_stop_source().stop_requested();
  }

  static RTP_MIDI_BOOL
  udp_callback(RTP_MIDI_SESSION *session, const void *remote_address,
	       size_t remote_address_length, uint16_t remote_port,
	       uint32_t address_flags, const unsigned char *packet,
               size_t packet_length, void *session_context);

  static void *new_peer_callback(RTP_MIDI_SESSION *session, RTP_MIDI_PEER *peer,
				 void *session_context);

  static void peer_update_callback(RTP_MIDI_SESSION *session,
				   RTP_MIDI_PEER *peer, uint32_t update_type,
				   void *session_context, void *peer_context);

  static void save_peer(RTP_MIDI_PEER *peer,
                        RtpMidiDeviceConfiguration &config);

  static void unsave_peer(RTP_MIDI_PEER *peer,
                          RtpMidiDeviceConfiguration &config);

private:
  // TODO: access to this config reference is not thread safe
  RtpMidiDeviceConfiguration &_config;

  RTP_MIDI_SESSION *_rtp_midi_session;
  std::jthread _rtp_clock_thread;

  int _ctrl_socket = -1;
  std::jthread _ctrl_recv_thread;

  int _data_socket = -1;
  std::jthread _data_recv_thread;

  std::map<RtpPeerKey, std::unique_ptr<RtpPeer>> _peers;
  std::jthread _mdns_thread;

  static uint64_t get_current_time();
};

} // namespace pc::midi
