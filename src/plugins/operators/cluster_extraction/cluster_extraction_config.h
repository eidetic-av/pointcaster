#pragma once

#include <config/output_value.h>
#include <plugins/backend/backend_types.h>
#include <pointcaster/point_cloud.h>
#include <rfl/DefaultVal.hpp>
#include <rfl/Literal.hpp>
#include <string>

namespace pc::operators {

class ClusterExtractionOperator;

struct ClusterExtractionConfiguration {
  std::string id;                      // @hidden
  rfl::DefaultVal<std::string> label;  // @hidden
  rfl::DefaultVal<bool> active = true; // @hidden

  // output fields
  Output<int> point_count = 0;
  Output<AabbListPtr> clusters;

  rfl::DefaultVal<BackendType> backend = BackendType::CPU;

  using OperatorType = ClusterExtractionOperator;
  using Tag = rfl::Literal<"clusterExtraction">;
  static constexpr auto PluginName = "ClusterExtractionOperator";
};

} // namespace pc::operators