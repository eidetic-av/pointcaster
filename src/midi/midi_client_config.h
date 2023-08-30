#pragma once

#include <map>
#include <serdepp/serde.hpp>
#include <serdepp/serializer.hpp>
#include <string>
#include <tuple>
#include <fmt/format.h>

namespace pc::midi {

struct MidiClientConfiguration {
  bool show_window = false;
  bool enable = true;

  // a map of maps, top layer midi port name, then cc_string -> slider_id bindings
  std::map<std::string, std::map<std::string, std::string>> cc_gui_bindings;

  DERIVE_SERDE(MidiClientConfiguration,
               (&Self::show_window, "show_window")(&Self::enable, "enable")
	       [attributes(make_optional)](&Self::cc_gui_bindings, "bindings"))
};

inline std::string cc_string(int32_t channel, int32_t control_num) {
  return fmt::format("{}:{}", channel, control_num);
}

} // namespace pc::midi
