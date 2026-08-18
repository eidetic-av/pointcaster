#pragma once

#include <config/core_types_reflection.h>
#include <config/output_value.h>
#include <plugins/backend/backend_types.h>
#include <pointcaster/core_types.h>
#include <rfl/DefaultVal.hpp>
#include <rfl/Literal.hpp>
#include <string>

namespace pc::operators {

class RangeFilterOperator;

struct RangeFilterConfiguration {
  std::string id;                      // @hidden
  rfl::DefaultVal<std::string> label;  // @hidden
  rfl::DefaultVal<bool> active = true; // @hidden

  rfl::DefaultVal<position_bounds> bounds = pc::default_config_bounds;

  Output<int> point_count = 0;

  rfl::DefaultVal<BackendType> backend = BackendType::CPU; // @hidden

  using OperatorType = RangeFilterOperator;
  using Tag = rfl::Literal<"rangeFilter">;
  static constexpr auto PluginName = "RangeFilterOperator";
};

} // namespace pc::operators
