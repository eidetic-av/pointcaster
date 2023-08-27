#include "midi_client.h"
#include "../logger.h"
#include <imgui.h>
#include <libremidi/client.hpp>
#include <libremidi/libremidi.hpp>
#include <map>
#include <memory>
#include <vector>

namespace pc {

std::shared_ptr<MidiClient> MidiClient::_instance;
std::mutex MidiClient::_mutex;

MidiClient::MidiClient(MidiClientConfiguration &config) : _config(config) {

  for (auto api : libremidi::available_apis()) {

    std::string api_name(libremidi::get_api_display_name(api));

    if (_midi_observers.contains(api_name))
      continue;

    // ignore sequencer inputs
    if (api_name.find("sequencer") != std::string::npos)
      continue;
    // ignore JACK for now
    if (api_name.find("JACK") != std::string::npos)
      continue;

    pc::logger->info("Initiliasing {} MIDI API", api_name);

    libremidi::observer_configuration observer_config{
        .input_added =
            [this, api](const auto &p) { this->handle_input_added(p, api); },
        .input_removed =
            [this](const auto &p) { this->handle_input_removed(p); }};

    _midi_observers.emplace(
        api_name,
        libremidi::observer{observer_config,
                            libremidi::observer_configuration_for(api)});
  }
  pc::logger->info("Waiting for MIDI hotplug events");
}

MidiClient::~MidiClient() {}

void MidiClient::handle_input_added(const libremidi::input_port &port,
                                    const libremidi::API &api) {
  auto [iter, emplaced] = _midi_inputs.emplace(
      port.port_name,
      libremidi::midi_in{{.on_message = std::bind(&MidiClient::handle_message,
                                                  this, std::placeholders::_1)},
                         libremidi::midi_in_configuration_for(api)});
  if (!emplaced)
    return;
  try {
    iter->second.open_port(port);
    pc::logger->info("{}: added input port {}", port.device_name,
                     port.port_name);
  } catch (libremidi::driver_error e) {
    pc::logger->warn("{}: {}", port.device_name, e.what());
  }
}

void MidiClient::handle_input_removed(const libremidi::input_port &port) {
  if (!_midi_inputs.contains(port.port_name))
    return;
  _midi_inputs.at(port.port_name).close_port();
  _midi_inputs.erase(port.port_name);
  pc::logger->info("{}: removed input port {}", port.device_name,
                   port.port_name);
}

void MidiClient::handle_message(const libremidi::message &msg) {
  using libremidi::message_type;
  auto channel = msg.get_channel();
  auto msg_type = msg.get_message_type();
  if (msg_type == message_type::CONTROL_CHANGE) {
    auto control_num = msg[1];
    auto value = msg[2];
    pc::logger->debug("received CC: {}.{}.{}", channel, control_num, value);
    return;
  }
  if (msg_type == message_type::PITCH_BEND) {
    unsigned int value = (msg[2] << 7) | msg[1];
    pc::logger->debug("received PB: {}.{}", channel, value);
    return;
  }
}

void MidiClient::draw_imgui_window() {
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("MIDI", nullptr);
  ImGui::Checkbox("Enable", &_config.enable);
  ImGui::End();
}

} // namespace pc
