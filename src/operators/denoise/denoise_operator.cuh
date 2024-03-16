#include "../../structs.h"
#include "denoise_operator.gen.h"
#include <thrust/functional.h>

namespace pc::operators {

__device__ bool DenoiseOperator::operator()(indexed_point_t point) const {
  auto &pos = thrust::get<0>(point);
  return true;
};

} // namespace pc::operators
