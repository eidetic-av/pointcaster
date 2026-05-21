#pragma once

#include <camera/camera_config.h>
#include <config/color_transform_config.h>
#include <plugins/backend/backend_types.h>
#include <rfl/Literal.hpp>
#include <string>

namespace pc::operators {

class FringeRemovalOperator;

struct FringeRemovalConfiguration {
  std::string id;     // @hidden
  bool active = true; // @hidden

  rfl::DefaultVal<BackendType> backend = BackendType::CPU;

  using OperatorType = FringeRemovalOperator;
  using Tag = rfl::Literal<"fringeRemoval">;
  static constexpr auto PluginName = "FringeRemovalOperator";
};

} // namespace pc::operators