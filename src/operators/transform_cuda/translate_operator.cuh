#include "../../structs.h"
#include "translate_operator.gen.h"
#include <thrust/functional.h>

namespace pc::operators::cuda {

using pc::types::color;
using pc::types::position;

__device__ indexed_point_t
TranslateOperator::operator()(indexed_point_t point) const {

  position pos = thrust::get<0>(point);
  const auto &offset = _config.offset;

  // mm to metres
  Float3 pos_m{(float)pos.x / 1000.0f, (float)pos.y / 1000.0f,
               (float)pos.z / 1000.0f};
  // translation
  Float3 new_pos{pos_m.x + offset.x, pos_m.y + offset.y, pos_m.z + offset.z};
  // back to mm
  position pos_out = {(short)__float2int_rd(new_pos.x * 1000),
                      (short)__float2int_rd(new_pos.y * 1000),
                      (short)__float2int_rd(new_pos.z * 1000), 0};

  return thrust::make_tuple(pos_out, thrust::get<1>(point),
                            thrust::get<2>(point));
};

} // namespace pc::operators::cuda
