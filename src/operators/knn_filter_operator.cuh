#include "../structs.h"
#include "knn_filter_operator.h"
#include <thrust/functional.h>

namespace pc::operators {

using pc::types::color;
using pc::types::position;

__device__ bool KNNFilterOperator::operator()(point_t point) const {

  position pos = thrust::get<0>(point);
  color col = thrust::get<1>(point);

  return pos.x > 0;
};

} // namespace pc::operators
