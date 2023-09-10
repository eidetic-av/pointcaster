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
  int last_value;
  int min = 0;
  int max = 127;

  DERIVE_SERDE(MidiBindingTarget,
               (&Self::id, "id")
	       (&Self::last_value, "last")
	       (&Self::min, "min")
	       (&Self::max, "max"))
};

struct MidiClientConfiguration {
  bool show_window = false;
  bool enable = true;

  float input_lerp = 0.65f;

  // a map of maps, top layer midi port name, then cc_string -> binding targets
  std::map<std::string, std::map<std::string, MidiBindingTarget>>
      cc_gui_bindings;

  DERIVE_SERDE(MidiClientConfiguration,
               (&Self::show_window, "show_window")(&Self::enable, "enable")(
                   &Self::input_lerp, "input_lerp")(&Self::cc_gui_bindings,
                                                    "bindings", make_optional))
};

inline std::string cc_string(int32_t channel, int32_t control_num) {
  return fmt::format("{}:{}", channel, control_num);
}

} // namespace pc::midi
