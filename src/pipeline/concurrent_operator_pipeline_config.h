#pragma once

#include <rfl/DefaultVal.hpp>

namespace pc::operators {

struct ConcurrentOperatorPipelineConfiguration {
  rfl::DefaultVal<int> update_hz = 120;
  rfl::DefaultVal<int> concurrency = 4;
};

} // namespace pc::operators