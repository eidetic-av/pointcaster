#pragma once

#include "../tween/tween_config.gen.h"
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

struct MidiOutputRoute {
  uint8_t channel;
  uint8_t cc;
  float last_value;
};

using MidiOutputRouteMap = std::map<std::string, MidiOutputRoute>;

struct MidiClientConfiguration {
  bool show_window = false;
  bool enable = true;
  float input_lerp = 0.65f;
  CCGuiBindings cc_gui_bindings; // @optional
  bool show_routes = true;
  MidiOutputRouteMap output_routes; // @optional
};
  // MidiOutputsConfiguration outputs;

inline std::string cc_string(int32_t channel, int32_t control_num) {
  return fmt::format("{}_{}", channel, control_num);
}

} // namespace pc::midi
