#pragma once
#include <rfl/DefaultVal.hpp>
#include <string>

namespace pc::devices {
struct DeviceGroup {
  std::string id;
  rfl::DefaultVal<std::string> label;
  rfl::DefaultVal<bool> active = true;
  rfl::DefaultVal<bool> render = true;
  rfl::DefaultVal<bool> collapsed = false;
  rfl::DefaultVal<std::string> parent_id;
  rfl::DefaultVal<int> order = 0;
};
} // namespace pc::devices