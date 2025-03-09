#include "../structs.h"
#include "range_filter_operator.gen.h"
#include <thrust/functional.h>

namespace pc::operators {

__device__ bool RangeFilterOperator::operator()(indexed_point_t point) const {
  auto &pos = thrust::get<0>(point);

  const auto &center = _config.transform.position;
  const auto &size = _config.transform.size;

  // config is in metres, point cloud is in mm
  const Float3 min{(center.x - size.x) * 1000, (center.y - size.y) * 1000,
                   (center.z - size.z) * 1000};
  const Float3 max{(center.x + size.x) * 1000, (center.y + size.y) * 1000,
                   (center.z + size.z) * 1000};

  return pos.x >= min.x && pos.x <= max.x && pos.y >= min.y && pos.y <= max.y &&
         pos.z >= min.z && pos.z <= max.z;

  return true;
};

__device__ bool MinMaxXComparator::operator()(indexed_point_t lhs,
                                              indexed_point_t rhs) const {
  auto &lhs_pos = thrust::get<0>(lhs);
  auto &rhs_pos = thrust::get<0>(rhs);
  return lhs_pos.x < rhs_pos.x;
}

__device__ bool MinMaxYComparator::operator()(indexed_point_t lhs,
                                              indexed_point_t rhs) const {
  auto &lhs_pos = thrust::get<0>(lhs);
  auto &rhs_pos = thrust::get<0>(rhs);
  return lhs_pos.y < rhs_pos.y;
}

__device__ bool MinMaxZComparator::operator()(indexed_point_t lhs,
                                              indexed_point_t rhs) const {
  auto &lhs_pos = thrust::get<0>(lhs);
  auto &rhs_pos = thrust::get<0>(rhs);
  return lhs_pos.z < rhs_pos.z;
}

} // namespace pc::operators
