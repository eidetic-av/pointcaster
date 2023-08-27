#pragma once

#include <nlohmann/json.hpp>

struct MidiClientConfiguration {
  bool enable = true;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MidiClientConfiguration, enable);
