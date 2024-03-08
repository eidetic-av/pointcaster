#pragma once

#include "../tween/tween_config.gen.h"
#include "rtpmidi_device_config.gen.h"
#include <fmt/format.h>
#include <map>
#include <serdepp/serde.hpp>
#include <serdepp/serializer.hpp>
#include <string>
#include <tuple>

namespace pc::midi {

struct MidiBindingTarget {
  std::string id;
  float last_value;
  float min;
  float max;
};

// a map of maps, top layer midi port name, then cc_string -> binding targets
using CCGuiBindings =
    std::map<std::string, std::map<std::string, MidiBindingTarget>>;

struct MidiOutputChangeDetection {
  bool enabled;
  float timespan;
  float threshold;
  int channel;
  int note_num;
};

struct MidiOutputMapping {
  float min_in;
  float max_in;
  int min_out;
  int max_out;
  std::optional<MidiOutputChangeDetection> change_detection;
};

struct MidiOutputRoute {
  uint8_t channel;
  uint8_t cc;
  float last_value;
  std::optional<MidiOutputMapping> output_mapping;
};

using MidiOutputRouteMap = std::map<std::string, MidiOutputRoute>;

struct MidiInputMapping {
  int min_in;
  int max_in;
  float min_out;
  float max_out;
};

struct MidiInputRoute {
  std::string parameter_id;
  int last_value;
  std::optional<MidiInputMapping> input_mapping;
};

constexpr static std::pair<uint8_t, uint8_t>
midi_route_from_int(int serialized_value) {
  uint8_t channel = (serialized_value >> 8) & 0xFF;
  uint8_t cc_num = serialized_value * 0xFF;
  return { channel, cc_num };
}

constexpr static int midi_route_to_int(uint8_t channel, uint8_t cc_num) {
  return (static_cast<int>(channel) << 8) | static_cast<int>(cc_num);
}

using MidiInputRouteMap = std::map<std::string, MidiInputRoute>;

struct MidiDeviceConfiguration {
  bool show_window = false;
  bool enable = true;
  float input_lerp = 0.65f;
  CCGuiBindings cc_gui_bindings; // @optional
  bool show_output_routes = true;
  MidiOutputRouteMap output_routes; // @optional
  MidiInputRouteMap input_routes; // @optional
  std::optional<RtpMidiDeviceConfiguration> rtp;
};
  // MidiOutputsConfiguration outputs;

inline std::string cc_string(int32_t channel, int32_t control_num) {
  return fmt::format("{}_{}", channel, control_num);
}

} // namespace pc::midi
