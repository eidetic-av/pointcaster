#pragma once

#include "../serialization.h"
#include "../structs.h"
#include <variant>
#include "transformer_types.h"
#include "noise_transformer.h"

namespace pc::transformers {

using TransformerConfigurationVariant = std::variant<NoiseTransformerConfiguration>;

struct TransformerConfiguration {
  bool enabled = true;

  std::vector<TransformerConfigurationVariant> transformers;

  DERIVE_SERDE(TransformerConfiguration,
               (&Self::enabled, "enabled")(&Self::transformers, "transformers"))

  using MemberTypes = pc::reflect::type_list<bool, std::vector<TransformerConfigurationVariant>>;
  static const std::size_t MemberCount = 2;
};

} // namespace pc::transformers
