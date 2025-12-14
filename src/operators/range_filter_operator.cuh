#include "../structs.h"
#include "range_filter_operator.gen.h"
#include <thrust/functional.h>

namespace pc::operators {

__device__ __forceinline__ bool RangeFilterOperator::operator()(indexed_point_t point) const {
  auto &pos = thrust::get<0>(point);
  bool inside = pos.x >= min_mm.x && pos.x <= max_mm.x && pos.y >= min_mm.y &&
                pos.y <= max_mm.y && pos.z >= min_mm.z && pos.z <= max_mm.z;
  return _config.invert ? !inside : inside;
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
