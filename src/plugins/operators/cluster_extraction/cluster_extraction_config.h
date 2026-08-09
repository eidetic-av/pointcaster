#pragma once

#include <plugins/backend/backend_types.h>
#include <rfl/DefaultVal.hpp>
#include <rfl/Literal.hpp>
#include <string>


namespace pc::operators {

class ClusterExtractionOperator;

struct ClusterExtractionConfiguration {
  std::string id;     // @hidden
  bool active = true; // @hidden

  rfl::DefaultVal<int> point_count = 0;

  rfl::DefaultVal<BackendType> backend = BackendType::CPU;

  using OperatorType = ClusterExtractionOperator;
  using Tag = rfl::Literal<"clusterExtraction">;
  static constexpr auto PluginName = "ClusterExtractionOperator";
};

} // namespace pc::operators