#include "../structs.h"
#include "sample_filter_operator.h"
#include <thrust/functional.h>

namespace pc::operators {

__device__ bool SampleFilterOperator::operator()(indexed_point_t point) const {
  return thrust::get<2>(point) % _config.sample_count == 0;
};

} // namespace pc::operators
