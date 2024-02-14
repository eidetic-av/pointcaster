#include "../structs.h"
#include "range_filter_operator.gen.h"
#include <thrust/functional.h>

namespace pc::operators {

__device__ bool RangeFilterOperator::operator()(indexed_point_t point) const {

  auto &pos = thrust::get<0>(point);

  auto &min = _config.aabb.min;
  auto &max = _config.aabb.max;

  return pos.x >= min.x && pos.x <= max.x && pos.y >= min.y && pos.y <= max.y &&
	 pos.z >= min.z && pos.z <= max.z;
};

} // namespace pc::operators
