#include "../logger.h"
#include "noise_operator.cuh"
#include "rotate_operator.cuh"
#include "rake_operator.cuh"
#include "session_operator_host.h"
#include <variant>

namespace pc::operators {

void SessionOperatorHost::run_operators(operator_in_out_t begin,
                                        operator_in_out_t end) const {

  for (auto &operator_config : _config.operators) {
    std::visit(
        [&begin, &end](auto &&config) {
          using T = std::decay_t<decltype(config)>;
          if constexpr (std::is_same_v<T, NoiseOperatorConfiguration>) {
            if (config.enabled)
              thrust::transform(begin, end, begin, NoiseOperator(config));
          }
          else if constexpr (std::is_same_v<T, RotateOperatorConfiguration>) {
            if (config.enabled)
              thrust::transform(begin, end, begin, RotateOperator(config));
          }
          else if constexpr (std::is_same_v<T, RakeOperatorConfiguration>) {
            if (config.enabled)
              thrust::transform(begin, end, begin, RakeOperator(config));
          }
          // You can handle other types here as well with further if constexpr
          // blocks
        },
        operator_config);
  }
};
} // namespace pc::operators
