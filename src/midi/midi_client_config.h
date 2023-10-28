#pragma once

#include "../tween/tween_config.h"
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

  DERIVE_SERDE(MidiBindingTarget,
               (&Self::id, "id")
	       (&Self::last_value, "last")
	       (&Self::min, "min")
	       (&Self::max, "max"))
};


struct MidiOutputsConfiguration {
  bool unfolded = false;

  DERIVE_SERDE(MidiOutputsConfiguration,
	       (&Self::unfolded, "unfolded")
	       );
};

struct MidiClientConfiguration {
  bool show_window = false;
  bool enable = true;

  // TODO consolidate the next two into a InputsConfig struct
  float input_lerp = 0.65f;
  // a map of maps, top layer midi port name, then cc_string -> binding targets
  std::map<std::string, std::map<std::string, MidiBindingTarget>>
      cc_gui_bindings;

  MidiOutputsConfiguration outputs{};

  DERIVE_SERDE(MidiClientConfiguration,
               (&Self::show_window, "show_window")
	       (&Self::enable, "enable")
	       (&Self::input_lerp, "input_lerp")
	       (&Self::outputs, "outputs")
	       (&Self::cc_gui_bindings, "bindings", make_optional))
};

inline std::string cc_string(int32_t channel, int32_t control_num) {
  return fmt::format("{}_{}", channel, control_num);
}

} // namespace pc::midi