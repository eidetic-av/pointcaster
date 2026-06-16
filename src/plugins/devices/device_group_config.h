#pragma once

#include "config/group_sequence_config.h"
#include "config/transform_config.h"
#include <rfl/DefaultVal.hpp>
#include <string>

namespace pc::devices {
struct DeviceGroupConfiguration {
  std::string id;                          // @hidden
  rfl::DefaultVal<std::string> label;      // @hidden
  rfl::DefaultVal<bool> active = true;     // @hidden
  rfl::DefaultVal<bool> render = true;     // @hidden
  rfl::DefaultVal<bool> collapsed = false; // @hidden
  rfl::DefaultVal<std::string> parent_id;  // @hidden
  rfl::DefaultVal<int> order = 0;          // @hidden
  rfl::DefaultVal<TransformConfiguration> transform;
  rfl::DefaultVal<GroupSequenceConfiguration> sequence;
};
} // namespace pc::devices