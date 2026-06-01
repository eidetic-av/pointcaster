#pragma once

#include <rfl/DefaultVal.hpp>

namespace pc::pipeline {

struct SessionOperatorPipelineConfiguration {
  rfl::DefaultVal<int> update_hz = 240;
  rfl::DefaultVal<int> concurrency = 4;
};

} // namespace pc::pipeline