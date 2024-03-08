#include "midi_device.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../math.h"
#include "../parameters.h"
#include "../publisher/publisher.h"
#include "../string_utils.h"
#include "../tween/tween_manager.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <imgui.h>
#include <libremidi/client.hpp>
#include <libremidi/libremidi.hpp>
#include <map>
#include <memory>
#include <tweeny/tweeny.h>
#include <vector>

namespace pc::midi {

using pc::tween::TweenManager;
using namespace pc::parameters;

static std::unordered_map<std::string, std::string> _port_keys;

MidiDevice::MidiDevice(MidiDeviceConfiguration &config)
    : _config(config), _sender_thread([this](auto st) { send_messages(st); }) {

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
        .input_added = [this,
                        api](const auto &p) { handle_input_added(p, api); },
        .input_removed = [this](const auto &p) { handle_input_removed(p); },
        .output_added = [this,
                         api](const auto &p) { handle_output_added(p, api); },
        .output_removed = [this](const auto &p) { handle_output_removed(p); }};

    _midi_observers.emplace(
        api_name,
        libremidi::observer{observer_config,
                            libremidi::observer_configuration_for(api)});
  }
  pc::logger->info("MIDI: Listening for MIDI hotplug events");

  // Create a virtual MIDI port
  _virtual_output.open_virtual_port("pointcaster 1");

  // And an RTP Midi (Apple Network Midi) port
  if (!_config.rtp.has_value()) {
    _config.rtp = RtpMidiDeviceConfiguration{};
  }
  _rtp_midi = std::make_unique<RtpMidiDevice>(*config.rtp);

  // for any MIDI cc bindings to UI parameters saved in the config,
  // notify the parameter its been 'bound'
  for (auto &[_, cc_bindings] : _config.cc_gui_bindings) {
    for (auto &[_, cc_binding] : cc_bindings) {
      parameter_states[cc_binding.id] = ParameterState::Bound;
      add_parameter_update_callback(
          cc_binding.id, [&cc_binding](auto, const auto &new_binding) {
            cc_binding.last_value = new_binding.current();
          });
      using parameter_type = std::decay_t<decltype(cc_binding.min)>;
      add_parameter_minmax_update_callback(
          cc_binding.id,
          [&cc_binding](const auto &old_binding, auto &new_binding) {
            cc_binding.min = std::get<parameter_type>(new_binding.min);
            cc_binding.max = std::get<parameter_type>(new_binding.max);
          });
      set_parameter_minmax(cc_binding.id, cc_binding.min, cc_binding.max);
      set_parameter_value(cc_binding.id, cc_binding.last_value, cc_binding.min,
                          cc_binding.max);
    }
  }

  declare_parameters("midi", _config);
  publisher::add(this);
}

MidiDevice::~MidiDevice() {
  _sender_thread.request_stop();
  _sender_thread.join();
  publisher::remove(this);
}

void MidiDevice::publish(libremidi::message_type message_type,
			 const uint8_t channel_num,
			 const uint8_t cc_or_note_num, const uint8_t value) {
  _messages_to_publish.enqueue(
      {message_type, channel_num, cc_or_note_num, value});
}

void MidiDevice::publish(const std::string_view topic,
                         const MidiOutputRoute &route) {
  const auto& value = route.last_value;
  const auto& channel_num = route.channel;
  const auto& cc_num = route.cc;

  uint8_t mapped_value;

  // if a custom midi output mapping has been specified, calculate the output
  // MIDI value
  if (route.output_mapping.has_value()) {
    const auto &m = *route.output_mapping;
    auto mapped_float =
	math::remap(m.min_in, m.max_in, static_cast<float>(m.min_out),
		    static_cast<float>(m.max_out), value, true);
    mapped_value = static_cast<uint8_t>(std::round(mapped_float));

    // if we're implementing change detection
    if (m.change_detection.has_value()) {
      static std::unordered_map<std::string, std::tuple<float, float>>
	  value_cache;

      using namespace std::chrono;
      static const auto start_time = duration_cast<milliseconds>(
	  high_resolution_clock::now().time_since_epoch());
      const auto now = duration_cast<milliseconds>(
	  high_resolution_clock::now().time_since_epoch());
      const float elapsed_seconds =
	  (now.count() - start_time.count()) / 1000.0f;

      auto topic_key = std::string{topic.data(), topic.size()};
      if (value_cache.contains(topic_key)) {
	const auto [cache_time, last_value] = value_cache[topic_key];

        bool change_detected = false;
	if (elapsed_seconds - cache_time >= m.change_detection->timespan) {
          change_detected = std::abs(mapped_float - last_value) >=
			    m.change_detection->threshold;
	  value_cache[topic_key] = {elapsed_seconds, mapped_float};
        }
        if (change_detected) {
	  const auto &cd = *m.change_detection;
	  _messages_to_publish.enqueue({libremidi::message_type::NOTE_ON,
					static_cast<uint8_t>(cd.channel),
					static_cast<uint8_t>(cd.note_num),
					127});
        }
      } else {
	value_cache[topic_key] = {elapsed_seconds, mapped_float};
      }
    }
  }
  // if no custom mapping is specified, map float from 0-1 to 0-127 midi
  else {
    mapped_value = static_cast<uint8_t>(
	std::round(std::max(std::min(value, 1.0f), 0.0f) * 127));
  }
  _messages_to_publish.enqueue({libremidi::message_type::CONTROL_CHANGE,
                                channel_num, cc_num, mapped_value});
}

void MidiDevice::send_messages(std::stop_token st) {
  using namespace std::chrono_literals;

  static constexpr auto thread_wait_time = 50ms;

  while (!_virtual_output.is_port_open() && !st.stop_requested()) {
    std::this_thread::sleep_for(thread_wait_time);
  }

  while (!st.stop_requested()) {
    MidiOutMessage message;
    while (_messages_to_publish.wait_dequeue_timed(message, thread_wait_time)) {
      auto status_byte =
	  static_cast<uint8_t>(message.type) + (message.channel_num - 1);
      _virtual_output.send_message(status_byte, message.cc_or_note_num,
				   message.value);
      if (_config.rtp->enable) {
	_rtp_midi->send_message(status_byte, message.cc_or_note_num,
				message.value);
      }
    }
  }

  pc::logger->info("Closed MIDI Client thread");
}

void MidiDevice::handle_input_added(const libremidi::input_port &port,
                                    const libremidi::API &api) {
  auto api_config = libremidi::midi_in_configuration_for(api);

  auto on_new_message =
      [this, port_name = port.port_name](const libremidi::message &msg) {
        handle_input_message(port_name, msg);
      };

  auto [iter, emplaced] = _midi_inputs.emplace(
      port.port_name,
      libremidi::midi_in{{.on_message = on_new_message}, api_config});
  if (!emplaced)
    return;

  try {
    iter->second.open_port(port);
    pc::logger->info("MIDI: '{}' added new input port '{}'", port.device_name,
                     port.port_name);
  } catch (libremidi::driver_error e) {
    pc::logger->warn("{}: {}", port.device_name, e.what());
  }
}

void MidiDevice::handle_input_removed(const libremidi::input_port &port) {
  if (!_midi_inputs.contains(port.port_name))
    return;
  _midi_inputs.at(port.port_name).close_port();
  _midi_inputs.erase(port.port_name);
  pc::logger->info("{}: removed input port {}", port.device_name,
                   port.port_name);
}

void MidiDevice::handle_input_message(const std::string &port_name,
                                      const libremidi::message &msg) {

  using libremidi::message_type;
  auto channel = msg.get_channel();
  auto msg_type = msg.get_message_type();
  auto &port_bindings = _config.cc_gui_bindings[port_name];

  if (msg_type == message_type::CONTROL_CHANGE) {
    auto control_num = msg[1];
    auto value = msg[2];
    auto cc = cc_string(channel, control_num);
    // pc::logger->debug("received CC: {}.{}.{}", channel, control_num, value);

    if (gui::learning_parameter) {
      gui::learning_parameter = false;
      auto [parameter_id, parameter_info] = gui::load_learning_parameter_info();

      struct ExistingBinding {
        std::string port_name;
        std::string cc_string;
        MidiBindingTarget midi_binding;
      };

      // first find out if we're either overwriting a controller binding,
      // or the control we're binding with is already bound to another parameter

      std::optional<ExistingBinding> overwrite_binding;
      std::optional<std::string> old_parameter_binding_id;

      for (auto &[port_name, cc_bindings] : _config.cc_gui_bindings) {
        for (auto &[cc_string, cc_binding] : cc_bindings) {
          if (cc_binding.id == parameter_id) {
            overwrite_binding = {port_name, cc_string, cc_binding};
          }
          if (cc_string == cc) {
            old_parameter_binding_id = cc_binding.id;
          }
        }
      }

      // if we're setting the same parameter to the same control, just revert
      // and exit here
      if (overwrite_binding.has_value() &&
          old_parameter_binding_id.has_value()) {
        auto new_id = overwrite_binding->midi_binding.id;
        auto old_id = old_parameter_binding_id.value();
        if (new_id == old_id) {
          parameter_states[old_id] = ParameterState::Bound;
          return;
        }
      }

      // if this parameter is already bound to MIDI,
      // overwrite it with the new MIDI control
      if (overwrite_binding.has_value()) {
        // transfer the old minmax values to the new binding
        parameter_info.min = overwrite_binding->midi_binding.min;
        parameter_info.max = overwrite_binding->midi_binding.max;
        // remove the callbacks for the old binding
        clear_parameter_callbacks(parameter_id);
        // and finally remove our cc mapping
        _config.cc_gui_bindings[overwrite_binding->port_name].erase(
            overwrite_binding->cc_string);
      }

      // and if this MIDI control is already bound to a different parameter,
      // erase that binding because it can't control more than one parameter
      // (for now)
      if (old_parameter_binding_id.has_value()) {
        unbind_parameter(old_parameter_binding_id.value());
      }

      // now create the new binding
      auto &cc_binding = port_bindings[cc];
      cc_binding.id = parameter_id;

      if (std::holds_alternative<float>(parameter_info.min)) {
        cc_binding.min = std::get<float>(parameter_info.min);
        cc_binding.max = std::get<float>(parameter_info.max);
      } else {
        cc_binding.min = std::get<int>(parameter_info.min);
        cc_binding.max = std::get<int>(parameter_info.max);
      }

      bind_parameter(cc_binding.id, parameter_info);

      add_parameter_update_callback(
          cc_binding.id, [&cc_binding](auto, const auto &new_binding) {
            cc_binding.last_value = new_binding.current();
          });

      if (std::holds_alternative<float>(parameter_info.min)) {
        add_parameter_minmax_update_callback(
            cc_binding.id,
            [&cc_binding](const auto &old_binding, auto &new_binding) {
              cc_binding.min = std::get<float>(new_binding.min);
              cc_binding.max = std::get<float>(new_binding.max);
            });
        set_parameter_minmax(cc_binding.id, std::get<float>(parameter_info.min),
                             std::get<float>(parameter_info.max));
      } else {
        add_parameter_minmax_update_callback(
            cc_binding.id,
            [&cc_binding](const auto &old_binding, auto &new_binding) {
              cc_binding.min = std::get<int>(new_binding.min);
              cc_binding.max = std::get<int>(new_binding.max);
            });
        set_parameter_minmax(cc_binding.id, std::get<int>(parameter_info.min),
                             std::get<int>(parameter_info.max));
      }

      set_parameter_value(cc_binding.id, static_cast<float>(value), 0.0f,
                          127.0f);

      add_parameter_erase_callback(cc_binding.id, [id = cc_binding.id] {
        TweenManager::instance()->remove(id);
      });

      gui::recording_result = ParameterState::Bound;

      return;
    }
    if (port_bindings.contains(cc)) {
      auto &binding = port_bindings[cc];
      using parameter_type = std::decay_t<decltype(binding.min)>;

      // just a simple linear interpolation between current and new midi
      // values... set using the TweenManager as a way to update the value every
      // frame until it's complete

      const auto set_value = [this, port_name, cc, binding,
                              target_midi_value =
                                  static_cast<float>(value)](auto, auto) {
        auto current_midi_value =
            math::remap(binding.min, binding.max, 0.0f, 127.0f,
                        get_parameter_value(binding.id));
        auto lerp_time = std::pow(std::min(_config.input_lerp, 0.9999f), 0.15f);
        auto new_midi_value =
            std::lerp(current_midi_value, target_midi_value, 1.0f - lerp_time);
        set_parameter_value(binding.id, new_midi_value, 0.0f, 127.0f);

        if (std::abs(new_midi_value - target_midi_value) < 0.001f) {
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
    if (gui::learning_parameter) {
      gui::learning_parameter = false;
      gui::recording_result = ParameterState::Unbound;
      return;
    }
    return;
  }
}

void MidiDevice::handle_output_added(const libremidi::output_port &port,
                                     const libremidi::API &api) {
  // auto api_config = libremidi::midi_out_configuration_for(api);

  // auto [iter, emplaced] = _midi_outputs.emplace(
  //     port.port_name,
  //     libremidi::midi_out{{}, api_config});
  // if (!emplaced) return;

  // try {
  //   iter->second.open_virtual_port("pointcaster output");
  //   pc::logger->info("MIDI: '{}' added new virtual output port '{}'",
  //   port.device_name,
  //                    port.port_name);
  // } catch (libremidi::driver_error e) {
  //   pc::logger->warn("{}: {}", port.device_name, e.what());
  // }
}

void MidiDevice::handle_output_removed(const libremidi::output_port &port) {
  // if (!_midi_outputs.contains(port.port_name)) return;
  // _midi_outputs.at(port.port_name).close_port();
  // _midi_outputs.erase(port.port_name);
  // pc::logger->info("{}: removed output port {}", port.device_name,
  // 		   port.port_name);
}

void MidiDevice::draw_imgui_window() {

  pc::gui::draw_parameters("midi", struct_parameters.at(std::string{"midi"}));

  bool &rtp_unfolded = _config.rtp.value().unfolded;
  ImGui::SetNextItemOpen(rtp_unfolded);
  if (ImGui::CollapsingHeader("Network MIDI")) {
    _rtp_midi->draw_imgui();
    rtp_unfolded = true;
  } else {
    rtp_unfolded = false;
  }

  ImGui::SetNextItemOpen(_config.show_output_routes);

  if (ImGui::CollapsingHeader("Output Routes##midi")) {
    _config.show_output_routes = true;

    if (ImGui::BeginTable("Routing", 5,
                          ImGuiTableFlags_SizingStretchProp |
                              ImGuiTableFlags_Resizable)) {

      std::optional<std::string> marked_for_delete;

      ImGui::TableSetupColumn("Route");
      ImGui::TableSetupColumn("Channel");
      ImGui::TableSetupColumn("CC");
      ImGui::TableSetupColumn("Last Value");
      ImGui::TableSetupColumn("##Delete");

      ImGui::TableHeadersRow();

      for (auto &route_kvp : _config.output_routes) {
        auto &key = route_kvp.first;
	auto &route = route_kvp.second;
	auto selected_route = _selected_output_route.has_value() &&
			      *_selected_output_route == route;

        auto param_id = std::to_string(gui::_parameter_index++);

        ImGui::TableNextRow();

        ImGui::TableNextColumn();

	ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(1.0f, 0.5f));
	if (ImGui::Selectable(key.c_str(), selected_route)) {
	  _selected_output_route = route;
	}
	ImGui::PopStyleVar();

        ImGui::TableNextColumn();

        constexpr auto inc_step = 1;

        ImGui::PushItemWidth(-1);
        if (ImGui::InputScalar(("##channel" + param_id).c_str(),
                               ImGuiDataType_U8, &route.channel, &inc_step)) {
          route.channel = std::clamp(route.channel, static_cast<uint8_t>(1),
                                     static_cast<uint8_t>(16));
        };
        ImGui::PopItemWidth();

        ImGui::TableNextColumn();

        ImGui::PushItemWidth(-1);
        if (ImGui::InputScalar(("##cc" + param_id).c_str(), ImGuiDataType_U8,
                               &route.cc, &inc_step)) {
          route.cc = std::clamp(route.cc, static_cast<uint8_t>(1),
                                static_cast<uint8_t>(128));
        };
        ImGui::PopItemWidth();

        ImGui::TableNextColumn();
        ImGui::Text("%f", route.last_value);

        ImGui::TableNextColumn();
        if (ImGui::Button(("x##" + param_id).c_str())) {
          marked_for_delete = key;
        };
      }
      ImGui::EndTable();

      if (_selected_output_route.has_value()) {

	ImGui::PushID(gui::_parameter_index++);

        auto &selected_route = (*_selected_output_route).get();

        if (selected_route.output_mapping.has_value()) {

          if (ImGui::BeginTable("##Selected output route", 2)) {

	    ImGui::TableSetupColumn("##mapping");
	    ImGui::TableSetupColumn("##change detection");

	    ImGui::TableNextRow();
	    ImGui::TableNextColumn();

	    static bool t = true;
	    ImGui::Checkbox("Remap", &t);

            auto &m = *selected_route.output_mapping;
            ImGui::InputFloat("Input min", &m.min_in);
            ImGui::InputFloat("Input max", &m.max_in);
            int min_out = static_cast<int>(m.min_out);
            int max_out = static_cast<int>(m.max_out);
            ImGui::InputInt("Output min", &min_out);
            ImGui::InputInt("Output max", &max_out);
            m.min_out = std::clamp(min_out, 0, 127);
            m.max_out = std::clamp(max_out, 0, 127);

            ImGui::TableNextColumn();

            auto &change_detection_conf =
                selected_route.output_mapping->change_detection;

            bool change_detection_initialised =
		change_detection_conf.has_value();
	    bool change_detection_enabled =
		change_detection_initialised && change_detection_conf->enabled;

            ImGui::Checkbox("Change detection", &change_detection_enabled);
	    change_detection_conf->enabled = change_detection_enabled;

            if (change_detection_enabled && !change_detection_initialised) {
              selected_route.output_mapping->change_detection =
                  MidiOutputChangeDetection{.timespan = 0.03f,
                                            .threshold = 1,
                                            .channel = selected_route.channel,
                                            .note_num = 1};
            }

            if (!change_detection_enabled) {
              ImGui::BeginDisabled();
            }

            float timespan{};
            float threshold{};
            int channel{};
            int note_num{};

            if (change_detection_initialised) {
	      timespan = change_detection_conf->timespan;
	      threshold = change_detection_conf->threshold;
	      channel = change_detection_conf->channel;
	      note_num = change_detection_conf->note_num;
	    }

            ImGui::InputFloat("Timespan", &timespan);
	    ImGui::InputFloat("Threshold", &threshold);
            ImGui::InputInt("Channel", &channel);
            ImGui::InputInt("Note", &note_num);

	    if (change_detection_initialised && change_detection_enabled) {
	      change_detection_conf->timespan =
		  std::clamp(timespan, 0.001f, 1.0f);
	      change_detection_conf->threshold = threshold;
              change_detection_conf->channel = std::clamp(channel, 0, 16);
              change_detection_conf->note_num = std::clamp(note_num, 0, 127);
	    }

            if (!change_detection_enabled) {
              ImGui::EndDisabled();
            }

            ImGui::EndTable();
          }

        } else {
          if (ImGui::Button("Remap selected route")) {
            selected_route.output_mapping = MidiOutputMapping{
                .min_in = 0.0f,
                .max_in = 1.0f,
                .min_out = 0,
                .max_out = 127,
            };
          }
        }

        ImGui::PopID();

        if (marked_for_delete.has_value()) {
          _config.output_routes.erase(marked_for_delete.value());
        }
      }
    } else {
      _config.show_output_routes = false;
    }
  }
}

} // namespace pc::midi
