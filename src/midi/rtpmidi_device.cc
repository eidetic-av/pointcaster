#include "rtpmidi_device.h"
#include "../gui/catpuccin.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../parameters.h"
#include "../string_utils.h"

#ifdef WIN32
#include <Winsock2.h>
#include <ws2tcpip.h>
typedef SSIZE_T ssize_t;
#else
#include <C-Binding/os_posx.c>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <mdns_cpp/logger.hpp>
#include <array>
#include <chrono>
#include <imgui.h>
#include <string_view>

namespace pc::midi {

RTP_MIDI_BOOL
RtpMidiDevice::udp_callback(RTP_MIDI_SESSION *session,
                            const void *remote_address,
                            size_t remote_address_length, uint16_t remote_port,
                            uint32_t address_flags, const unsigned char *packet,
                            size_t packet_length, void *session_context) {

  auto *device = reinterpret_cast<RtpMidiDevice *>(session_context);

  sockaddr_in dest{.sin_family = AF_INET, .sin_port = htons(remote_port)};
  memcpy(&dest.sin_addr, remote_address, remote_address_length);

  int sent_bytes;

  if ((address_flags & RTP_MIDI_ADDRESS_MASK_PORT) ==
      RTP_MIDI_ADDRESS_FLAG_DATA) {
#ifdef WIN32
    sent_bytes =
        sendto(device->data_socket(), reinterpret_cast<const char*>(packet), packet_length, 0,
               reinterpret_cast<const sockaddr *>(&dest), sizeof(dest));
#else
    sent_bytes =
        sendto(device->data_socket(), packet, packet_length, 0,
               reinterpret_cast<const sockaddr *>(&dest), sizeof(dest));
#endif
  } else {
#ifdef WIN32
    sent_bytes =
        sendto(device->ctrl_socket(), reinterpret_cast<const char*>(packet), packet_length, 0,
               reinterpret_cast<const sockaddr *>(&dest), sizeof(dest));
#else
    sent_bytes =
        sendto(device->ctrl_socket(), packet, packet_length, 0,
               reinterpret_cast<const sockaddr *>(&dest), sizeof(dest));
#endif
  }

  if (sent_bytes != static_cast<ssize_t>(packet_length)) {
    pc::logger->error("Failed to send on libRtpMidi UDP socket");
    // Log the error here
    return RTP_MIDI_FALSE;
  }

  return RTP_MIDI_TRUE;
}

void RtpMidiDevice::midi_callback(RTP_MIDI_SESSION *session,
				  uint32_t delta_time,
				  const unsigned char *command, size_t length,
				  RTP_MIDI_BOOL with_running_status,
				  void *session_context) {
  auto *device = reinterpret_cast<RtpMidiDevice *>(session_context);
  std::vector<unsigned char> buffer(length);
  std::memcpy(buffer.data(), command, length);
  device->on_receive(buffer);
}

void RtpMidiDevice::peer_update_callback(RTP_MIDI_SESSION *session,
					 RTP_MIDI_PEER *peer,
					 uint32_t update_type,
					 void *session_context,
					 void *peer_context) {
  RtpPeer *target_peer = reinterpret_cast<RtpPeer *>(peer_context);
  if (update_type == RTP_MIDI_PEER_UPDATE_STATE) {
    auto state = rtpMidiPeerState(peer);
    if (target_peer->connected && state > RTP_MIDI_PEER_STATE_ACTIVE) {
      rtpMidiSessionPeerRemove(session, peer);
      target_peer->peer = nullptr;
    }
    target_peer->connected = state == RTP_MIDI_PEER_STATE_ACTIVE;
  }
}

std::tuple<std::string, std::string, uint16_t>
get_peer_info(RTP_MIDI_PEER *peer) {
  auto *peer_context = rtpMidiPeerContext(peer);
  if (peer_context != nullptr) {
    auto *rtp_peer = reinterpret_cast<RtpPeer *>(peer_context);
    return {rtp_peer->name, rtp_peer->address, rtp_peer->port};
  }
  const char *name = rtpMidiPeerName(peer);
  const void *remote_address;
  size_t address_length;
  uint16_t port;
  uint32_t flags;
  if (!rtpMidiPeerRemoteAddress(peer, &remote_address, &address_length, &port,
				&flags)) {
    throw std::runtime_error(
	"Failed to obtain remote address from RtpMidi peer");
  };
  char ip_str[INET_ADDRSTRLEN];
  std::memset(ip_str, 0, INET_ADDRSTRLEN);
  if (inet_ntop(AF_INET, remote_address, ip_str, INET_ADDRSTRLEN) == nullptr) {
    throw std::runtime_error("Failed to convert remote address format");
  }
  return {std::string{name}, std::string{ip_str}, port};
};

void *RtpMidiDevice::new_peer_callback(RTP_MIDI_SESSION *session,
				       RTP_MIDI_PEER *peer,
				       void *session_context) {
  auto [name, ip, port] = get_peer_info(peer);
  auto *device = reinterpret_cast<RtpMidiDevice *>(session_context);
  RtpPeer *result = device->get_or_create_peer(name, ip, port);
  result->peer = peer;
  save_peer(peer, device->_config);
  return result;
}

uint64_t RtpMidiDevice::get_current_time() {
  using namespace std::chrono;
  const auto elapsed = duration_cast<microseconds>(
      high_resolution_clock::now().time_since_epoch());
  // rtp midi uses current time in 0.1msec ticks
  return elapsed.count() / 100;
}

void RtpMidiDevice::mdns_reply_callback(const std::string &msg) {

  if (msg.find("SRV") != std::string::npos) {

    // extract the IP address
    auto ip_end_pos = msg.find(':');
    std::string ip_address = msg.substr(0, ip_end_pos);

    // extract the application service name
    auto end_pos = msg.find("._apple-midi");
    auto temp_string = msg.substr(0, end_pos);
    auto start_pos = temp_string.rfind(' ') + 1;
    std::string name = msg.substr(start_pos, end_pos - start_pos);

    // extract the port
    start_pos = msg.rfind(' ') + 1;
    uint16_t port = static_cast<uint16_t>(std::stoi(msg.substr(start_pos)));

    pc::logger->debug("[mDNS] Found {} ({}:{})", name, ip_address, port);

    // create an (unlinked) RtpPeer for this host
    RtpPeer *rtp_peer = get_or_create_peer(name, ip_address, port);

    RtpPeerKey peer_key{name, ip_address, port};

    if (_config.enable && !rtp_peer->connected) {
      // if we have a serialized config entry for this discovered peer, then
      // automatically connect to it
      auto it = std::find(_config.peers.begin(), _config.peers.end(), peer_key);
      if (it != _config.peers.end()) {
        rtp_peer->connect(_rtp_midi_session);
      }
    }
  }
}

RtpMidiDevice::RtpMidiDevice(RtpMidiDeviceConfiguration &config)
    : _config(config) {

  using namespace std::chrono;
  using namespace std::chrono_literals;

  // Rtp midi initialisation and session setup

  constexpr static auto log =
      [](RTP_MIDI_SESSION *session, RTP_MIDI_PEER *peer, const char *message,
         uint32_t log_level, const unsigned char *data, size_t data_length,
         void *session_context, void *peer_session_context) {
        // pc::logger->debug("[rtp] {}", message);
      };

  if (!rtpMidiInitialize(get_current_time, log, RTP_MIDI_LOGGING_LEVEL_INFO,
                         0)) {
    pc::logger->error("Failed to initialise libRtpMidi");
    return;
  }

  _rtp_midi_session = rtpMidiSessionCreate(
      "pointcaster", 0, udp_callback, midi_callback, peer_update_callback,
      new_peer_callback, RTP_MIDI_LOGGING_LEVEL_INFO, 0, this);

  _rtp_clock_thread = std::jthread([this](auto st) {
    while (!st.stop_requested()) {
      rtpMidiSessionTick(_rtp_midi_session, get_current_time());
      std::this_thread::sleep_for(milliseconds(rtp_tick_time_ms));
    }
  });

  // UDP Socket setup and bind
  _ctrl_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (_ctrl_socket < 0) {
	  pc::logger->error("Failed to create RtpMidi control socket on port {}. Is another service running on this port?",
		  rtp_midi_ctrl_port);
    return;
  }
  sockaddr_in ctrl_address = {.sin_family = AF_INET,
			      .sin_port = htons(rtp_midi_ctrl_port)};
  if (inet_pton(AF_INET, _config.ip.data(), &ctrl_address.sin_addr) <= 0) {
    pc::logger->error("Failed to convert network address to binary format");
    return;
  }
  if (bind(_ctrl_socket, reinterpret_cast<sockaddr *>(&ctrl_address),
           sizeof(ctrl_address)) < 0) {
    pc::logger->error("Failed to bind RtpMidi control socket to port {}",
                      rtp_midi_ctrl_port);
    return;
  }

  try {
	  _data_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  }
  catch (std::exception& e){
	  pc::logger->error("Failed to create RtpMidi control socket on port {}. Is another service running on this port?",
		  rtp_midi_ctrl_port);
  }
  if (_data_socket < 0) {
    pc::logger->error("Failed to create RtpMidi control socket");
    return;
  }

  sockaddr_in data_address = {.sin_family = AF_INET,
                              .sin_port = htons(rtp_midi_data_port)};
  if (inet_pton(AF_INET, config.ip.data(), &data_address.sin_addr) <= 0) {
    pc::logger->error("Failed to convert network address to binary format");
    return;
  }
  if (bind(_data_socket, reinterpret_cast<sockaddr *>(&data_address),
           sizeof(data_address)) < 0) {
    pc::logger->error("Failed to bind RtpMidi data socket to port {}",
                      rtp_midi_data_port);
    return;
  }

  constexpr timeval recv_timeout{.tv_sec = 0, .tv_usec = 200000};
  if (setsockopt(_ctrl_socket, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char *>(&recv_timeout),
                 sizeof(recv_timeout)) < 0) {
    pc::logger->error("Failed to set ctrl socket timeout");
    throw std::exception();
  }
  if (setsockopt(_data_socket, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char *>(&recv_timeout),
                 sizeof(recv_timeout)) < 0) {
    pc::logger->error("Failed to set data socket timeout");
    throw std::exception();
  }

  // Setup received UDP message handling

  _ctrl_recv_thread = std::jthread([this](auto st) {
	  constexpr auto udp_recv_buffer_size = 9000;
	  unsigned char recv_buffer[udp_recv_buffer_size + 1];
	  sockaddr_in source_address;
	  socklen_t source_address_size;

	  while (!st.stop_requested()) {
		  // wait for data with a timeout
		  fd_set read_fds;
		  FD_ZERO(&read_fds);
		  FD_SET(_ctrl_socket, &read_fds);
		  const struct timeval timeout {
			 .tv_sec = 0,
			 .tv_usec = 50000 // 50ms
          };

          int result = select(0, &read_fds, nullptr, nullptr, &timeout);
		  if (result > 0) {
			  source_address_size = sizeof(source_address);
			  ssize_t recv_size = recvfrom(
				  _ctrl_socket, (char*)recv_buffer, sizeof(recv_buffer), 0,
				  reinterpret_cast<sockaddr*>(&source_address), &source_address_size);
			  if (recv_size > 0) {

				  rtpMidiSessionReceivedUdpPacket(
					  _rtp_midi_session, &source_address.sin_addr, sizeof(struct in_addr),
					  ntohs(source_address.sin_port),
					  RTP_MIDI_ADDRESS_FLAG_CTRL | RTP_MIDI_ADDRESS_FLAG_IPV4,
					  recv_buffer, recv_size);

			  }
			  else if (errno != EAGAIN && errno != EWOULDBLOCK) {
				  pc::logger->warn("Unhandled UDP recv error");
			  }
		  }
	  }
  });

  _data_recv_thread = std::jthread([this](auto st) {
    constexpr auto udp_recv_buffer_size = 9000;
    unsigned char recv_buffer[udp_recv_buffer_size + 1];
    sockaddr_in source_address;
    socklen_t source_address_size;

    while (!st.stop_requested()) {

		// wait for data with a timeout
		fd_set read_fds;
		FD_ZERO(&read_fds);
		FD_SET(_ctrl_socket, &read_fds);
		const struct timeval timeout {
			.tv_sec = 0,
				.tv_usec = 50000 // 50ms
		};

		int result = select(0, &read_fds, nullptr, nullptr, &timeout);
		if (result > 0) {
			source_address_size = sizeof(source_address);
			ssize_t recv_size = recvfrom(
				_data_socket, (char*)recv_buffer, sizeof(recv_buffer), 0,
				reinterpret_cast<sockaddr*>(&source_address), &source_address_size);
			if (recv_size > 0) {

				rtpMidiSessionReceivedUdpPacket(
					_rtp_midi_session, &source_address.sin_addr, sizeof(struct in_addr),
					ntohs(source_address.sin_port),
					RTP_MIDI_ADDRESS_FLAG_DATA | RTP_MIDI_ADDRESS_FLAG_IPV4,
					recv_buffer, recv_size);

			}
			else if (errno != EAGAIN && errno != EWOULDBLOCK) {
				pc::logger->warn("Unhandled UDP recv error");
			}
		}
    }
  });

  pc::logger->info("Listening for RtpMidi messages on UDP ports {},{}",
                   rtp_midi_ctrl_port, rtp_midi_data_port);

  // look for peers responding to mdns/avahi/bonjour
  mdns_cpp::Logger::setLoggerSink(
      [this](auto &msg) { mdns_reply_callback(msg); });

  _mdns_query = std::async(std::launch::async, [this] {
    mdns_cpp::mDNS mdns;
    mdns.executeQuery("_apple-midi._udp.local");
  });

  parameters::declare_parameters("workspace", "rtpmidi", _config);
}

RtpMidiDevice::~RtpMidiDevice() {
  for (auto &peer_entry : _peers) {
    auto &peer = peer_entry.second;
    peer->disconnect(_rtp_midi_session);
  }
  if (_rtp_clock_thread.joinable()) {
    _rtp_clock_thread.request_stop();
    _rtp_clock_thread.join();
  }

  rtpMidiSessionFree(_rtp_midi_session);
  rtpMidiFinalize();

  if (_ctrl_recv_thread.joinable()) {
    _ctrl_recv_thread.request_stop();
    _ctrl_recv_thread.join();
  }
  if (_data_recv_thread.joinable()) {
    _data_recv_thread.request_stop();
    _data_recv_thread.join();
  }

#ifdef WIN32
  bool do_cleanup = false;
  if (_ctrl_socket != INVALID_SOCKET) {
	  closesocket(_ctrl_socket);
      do_cleanup = true;
  }
  if (_data_socket != INVALID_SOCKET) {
	  closesocket(_data_socket);
      do_cleanup = true;
  }
  if (do_cleanup) {
    WSACleanup();
  }
#else
  if (_ctrl_socket >= 0) {
	  close(_ctrl_socket);
  }
  if (_data_socket >= 0) {
	  close(_data_socket);
  }
#endif
};

RtpPeer *RtpMidiDevice::get_or_create_peer(std::string_view name,
					   std::string_view address,
					   uint16_t port) {
  auto [it, _] = _peers.emplace(
      RtpPeerKey{
	  {name.data(), name.size()}, {address.data(), address.size()}, port},
      std::make_unique<RtpPeer>(name.data(), address.data(), port, this));
  return it->second.get();
}

void RtpMidiDevice::save_peer(RTP_MIDI_PEER *peer,
                              RtpMidiDeviceConfiguration &config) {
  auto [name, ip, port] = get_peer_info(peer);
  RtpPeerKey peer_key{std::move(name), std::move(ip), port};
  auto it = std::find(config.peers.begin(), config.peers.end(), peer_key);
  if (it == config.peers.end()) {
    config.peers.emplace_back(std::move(peer_key));
  }
}

void RtpMidiDevice::unsave_peer(RTP_MIDI_PEER *peer,
				RtpMidiDeviceConfiguration &config) {
  auto [name, ip, port] = get_peer_info(peer);
  RtpPeerKey peer_key{std::move(name), std::move(ip), port};
  auto it = std::find(config.peers.begin(), config.peers.end(), peer_key);
  if (it != config.peers.end()) {
    config.peers.erase(it);
  }
}

void RtpMidiDevice::draw_imgui() {

  using namespace ImGui;

  pc::gui::draw_parameters(
      "rtpmidi", parameters::struct_parameters.at(std::string{"rtpmidi"}));

  static bool was_enabled = _config.enable;
  bool enabled = _config.enable;

  // this will disconnect or reconnect peers that are part of our
  // serialized config when the "enable" button is toggled
  bool should_disconnect_peers = was_enabled && !enabled;
  bool should_reconnect_peers = !was_enabled && enabled;

  if (!enabled) {
    BeginDisabled();
  }

  Dummy({10, 0});
  SameLine();

  if (BeginListBox("Discovered peers")) {
    if (_peers.empty()) {
      BeginDisabled();
      Selectable("None");
      EndDisabled();
    } else {

      std::optional<RtpPeer> selected_peer;

      Dummy({0, 3});
      for (auto &entry : _peers) {
        const auto &rtp_peer = entry.second;
        auto peer_label =
            fmt::format(" {} ({})", rtp_peer->name, rtp_peer->address);
        SameLine();
        Selectable(peer_label.c_str());

	// this will disconnect or reconnect peers that are part of our
	// serialized config when the "enable" button is toggled

        if (should_reconnect_peers || should_disconnect_peers) {
          RtpPeerKey peer_key{rtp_peer->name, rtp_peer->address,
			      rtp_peer->port};
	  auto it =
	      std::find(_config.peers.begin(), _config.peers.end(), peer_key);
	  if (it != _config.peers.end()) {
	    if (should_disconnect_peers) {
	      rtp_peer->disconnect(_rtp_midi_session);
	    } else if (should_reconnect_peers) {
	      rtp_peer->connect(_rtp_midi_session);
	    }
	  }
        }

        // this will disconnect or reconnect peers when they are double-clicked

        if (IsItemHovered() && IsMouseDoubleClicked(0)) {
          if (!rtp_peer->connected) {

            // save peer to our serialized configuration if we manually
            // connect to it
            static constexpr auto on_manual_connect = [](auto *peer,
                                                         auto *session) {
              auto *device = reinterpret_cast<RtpMidiDevice *>(
                  rtpMidiSessionContext(session));
              save_peer(peer, device->_config);
            };

            rtp_peer->connect(_rtp_midi_session, on_manual_connect);

          } else {

            // remove peer from our serialized configuration if we manually
            // disconnect from
            static constexpr auto on_manual_disconnect = [](auto *peer,
                                                            auto *session) {
              auto *device = reinterpret_cast<RtpMidiDevice *>(
                  rtpMidiSessionContext(session));
              unsave_peer(peer, device->_config);
            };

            if (rtp_peer->peer == nullptr) {
              pc::logger->error("Disconnecting from nullptr peer");
            }

            rtp_peer->disconnect(_rtp_midi_session, on_manual_disconnect);
          }
        }
        if (rtp_peer->connected) {
          SameLine();
          using namespace catpuccin::imgui;
          PushStyleColor(ImGuiCol_Text, mocha_blue);
          Bullet();
          PopStyleColor();
	  NewLine();
        }
        NewLine();
      }
    }
    EndListBox();
  }

  if (!enabled) {
    EndDisabled();
  }

  Dummy({0, 10});

  was_enabled = _config.enable;
}

} // namespace pc::midi
