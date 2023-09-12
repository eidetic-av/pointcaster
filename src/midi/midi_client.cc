#include "midi_client.h"
#include "../gui_helpers.h"
#include "../logger.h"
#include "../string_utils.h"
#include "../tween/tween_manager.h"
#include <cmath>
#include <imgui.h>
#include <libremidi/client.hpp>
#include <libremidi/libremidi.hpp>
#include <map>
#include <memory>
#include <tweeny/tweeny.h>
#include <vector>

namespace pc::midi {

using pc::tween::TweenManager;

std::shared_ptr<MidiClient> MidiClient::_instance;
std::mutex MidiClient::_mutex;

static std::unordered_map<std::string, std::string> _port_keys;

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

    pc::logger->info("MIDI: Initiliasing {} MIDI API", api_name);

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
  pc::logger->info("MIDI: Waiting for MIDI hotplug events");

  // for any MIDI cc bindings to UI sliders saved in the config,
  // notify the slider its been 'bound'
  for (auto &[_, cc_bindings] : _config.cc_gui_bindings) {
    for (auto &[_, cc_binding] : cc_bindings) {
      gui::slider_states[cc_binding.id] = gui::SliderState::Bound;
      gui::add_slider_update_callback(
	  cc_binding.id, [&cc_binding](const auto &slider_binding) {
	    cc_binding.last_value = slider_binding.value;
	  });
      gui::add_slider_minmax_update_callback(
          cc_binding.id, [&cc_binding](const auto &slider_binding) {
	    cc_binding.min = slider_binding.min;
	    cc_binding.max = slider_binding.max;
	    std::cout << "set " << cc_binding.min << "to " << slider_binding.min << "\n";
          });
      gui::set_slider_minmax(cc_binding.id, cc_binding.min, cc_binding.max);
      gui::set_slider_value(cc_binding.id, cc_binding.last_value,
			    cc_binding.min, cc_binding.max);
    }
  }
}

MidiClient::~MidiClient() {}

void MidiClient::handle_input_added(const libremidi::input_port &port,
                                    const libremidi::API &api) {
  auto api_config = libremidi::midi_in_configuration_for(api);

  auto on_new_message =
      [this, port_name = port.port_name](const libremidi::message &msg) {
        handle_message(port_name, msg);
      };

  auto [iter, emplaced] = _midi_inputs.emplace(
      port.port_name,
      libremidi::midi_in{{.on_message = on_new_message}, api_config});
  if (!emplaced) return;

  try {
    iter->second.open_port(port);
    pc::logger->info("MIDI: '{}' added new input port '{}'", port.device_name,
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

void MidiClient::handle_message(const std::string &port_name,
                                const libremidi::message &msg) {

  using libremidi::message_type;
  auto channel = msg.get_channel();
  auto msg_type = msg.get_message_type();

  if (msg_type == message_type::CONTROL_CHANGE) {
    auto control_num = msg[1];
    auto value = msg[2];
    auto cc = cc_string(channel, control_num);
    // pc::logger->debug("received CC: {}.{}.{}", channel, control_num, value);

    if (gui::recording_slider) {
      gui::recording_slider = false;
      auto& cc_binding = _config.cc_gui_bindings[port_name][cc];
      cc_binding.id = gui::load_recording_slider_id();
      gui::add_slider_update_callback(
	  cc_binding.id, [&cc_binding](const auto &slider_binding) {
	    cc_binding.last_value = slider_binding.value;
	  });
      gui::add_slider_minmax_update_callback(
          cc_binding.id, [&cc_binding](const auto &slider_binding) {
	    cc_binding.min = slider_binding.min;
	    cc_binding.max = slider_binding.max;
	    std::cout << "set " << cc_binding.min << "to " << slider_binding.min << "\n";
          });
      gui::set_slider_value(cc_binding.id, static_cast<float>(value), 0.0f,
			    127.0f);
      gui::recording_result = gui::SliderState::Bound;
      return;
    }
    if (_config.cc_gui_bindings[port_name].contains(cc)) {

      auto &binding = _config.cc_gui_bindings[port_name][cc];

      // just a simple linear interpolation between current and new values...
      // set using the TweenManager as a way to update the value every frame
      // until it's complete

      const auto set_value = [this, port_name, cc, binding,
                              target_value = value](auto _, auto __) {
        auto current_value = gui::get_slider_value(binding.id);

        auto lerp_time = std::pow(std::min(_config.input_lerp, 0.9999f), 0.15f);
        auto new_value =
            std::lerp(current_value, target_value, 1.0f - lerp_time);

	gui::set_slider_value(binding.id, new_value, 0.0f, 127.0f);

        if (std::abs(new_value - target_value) < 0.001f) {
          TweenManager::instance()->remove(binding.id);
        }
        return false;
      };

      auto tween = tweeny::from(0.0f)
                       .to(1.0f)
                       .onStep(set_value)
                       .during(120000) // 120 seconds max
                       .via(tweeny::easing::linear);
      TweenManager::instance()->add(binding.id, tween);
    }
    return;
  }
  if (msg_type == message_type::PITCH_BEND) {
    unsigned int value = (msg[2] << 7) | msg[1];
    // pc::logger->debug("received PB: {}.{}", channel, value);
    if (gui::recording_slider) {
      gui::recording_slider = false;
      gui::recording_result = gui::SliderState::Unbound;
      return;
    }
    return;
  }
}

void MidiClient::draw_imgui_window() {
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("MIDI", nullptr);

  gui::draw_slider("Input Lerp", &_config.input_lerp, 0.0f, 1.0f, 0.65f);

  ImGui::End();
}

} // namespace pc::midi
