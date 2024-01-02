#include "../structs.h"
#include "rotate_operator.h"
#include <thrust/functional.h>
#include <Eigen/Geometry>

namespace pc::operators {

using pc::types::color;
using pc::types::position;

__host__ __device__ static inline float as_rad(float deg) {
  constexpr auto mult = 3.141592654f / 180.0f;
  return deg * mult;
};

__device__ indexed_point_t RotateOperator::operator()(indexed_point_t point) const {

  position pos = thrust::get<0>(point);
  color col = thrust::get<1>(point);

  using namespace Eigen;

  Vector3f pos_f {(float)pos.x, (float)pos.y, (float)pos.z};

  auto& euler_angles = _config.euler_angles;

  AngleAxisf rot_x(as_rad(euler_angles.x), Vector3f::UnitX());
  AngleAxisf rot_y(as_rad(euler_angles.y), Vector3f::UnitY());
  AngleAxisf rot_z(as_rad(euler_angles.z), Vector3f::UnitZ());
  Quaternionf q = rot_z * rot_y * rot_x;

  pos_f = q * pos_f;

  position pos_out = { (short)__float2int_rd(pos_f.x()),
					  (short)__float2int_rd(pos_f.y()),
					  (short)__float2int_rd(pos_f.z()), 0 };

  return thrust::make_tuple(pos_out, col, thrust::get<2>(point));
};

} // namespace pc::operators
