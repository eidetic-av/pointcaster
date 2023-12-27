#pragma once

#include "../serialization.h"
#include "../structs.h"
#include "noise_operator.h"
#include "rotate_operator.h"
#include "rake_operator.h"
#include "operator_types.h"
#include <variant>

namespace pc::operators {

using OperatorConfigurationVariant = std::variant<
	NoiseOperatorConfiguration,
	RotateOperatorConfiguration,
	RakeOperatorConfiguration>;

struct OperatorHostConfiguration {
  bool enabled = true;

  std::vector<OperatorConfigurationVariant> operators;

  DERIVE_SERDE(OperatorHostConfiguration,
	       (&Self::enabled, "enabled")
	       (&Self::operators, "operators"))

  using MemberTypes = pc::reflect::type_list<bool, std::vector<OperatorConfigurationVariant>>;
  static const std::size_t MemberCount = 2;
};

} // namespace pc::operators
