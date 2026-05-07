#pragma once

#include <config/color_transform_config.h>
#include <rfl/Literal.hpp>
#include <string>

namespace pc::operators {

class FringeRemovalOperator;

struct FringeRemovalConfiguration {
  std::string id;     // @hidden
  bool active = true; // @hidden

  using OperatorType = FringeRemovalOperator;
  using Tag = rfl::Literal<"fringeRemoval">;
  static constexpr auto PluginName = "FringeRemovalOperator";
};

} // namespace pc::operators