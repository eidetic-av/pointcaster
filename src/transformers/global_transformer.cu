#include "../logger.h"
#include "global_transformer.h"
#include "noise_transformer.cuh"
#include <variant>

namespace pc::transformers {

void GlobalTransformer::run_transformers(transformer_in_out_t begin,
				      transformer_in_out_t end) const {

  for (auto &transformer_config : _config.transformers) {
    std::visit(
	[&begin, &end](auto &&config) {
	  using T = std::decay_t<decltype(config)>;
	  if constexpr (std::is_same_v<T, NoiseTransformerConfiguration>) {
	    if (config.enabled)
	      thrust::transform(begin, end, begin, NoiseTransformer(config));
	  }
	  // You can handle other types here as well with further if constexpr
	  // blocks
	},
	transformer_config);
  }
};
} // namespace pc::transformers
