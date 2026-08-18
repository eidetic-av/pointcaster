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

  rfl::DefaultVal<bool> invert = false;
  rfl::DefaultVal<bool> bypass = false;
  rfl::DefaultVal<int> count_threshold = 5; // @minmax(0, 1000)
  rfl::DefaultVal<int> max_fill = 5000;     // @minmax(1, 100000)

  Output<int> point_count = 0;
  Output<float> fill_value = 0;
  Output<float> proportion = 0;
  Output<position_bounds> occupied_bounds;

  rfl::DefaultVal<BackendType> backend = BackendType::CPU;

  using OperatorType = RangeFilterOperator;
  using Tag = rfl::Literal<"rangeFilter">;
  static constexpr auto PluginName = "RangeFilterOperator";
};

} // namespace pc::operators
