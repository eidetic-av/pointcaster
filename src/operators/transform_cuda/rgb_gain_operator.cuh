#include "../../structs.h"
#include "rgb_gain_operator.gen.h"
#include <thrust/functional.h>

namespace pc::operators::cuda {

using pc::types::color;
using pc::types::Float3;
using pc::types::position;

__device__ indexed_point_t
RGBGainOperator::operator()(indexed_point_t point) const {
  const color in_col = thrust::get<1>(point);
  const Float3 gain = _config.gain;
  const float multiplier = _config.multiplier;
  const Float3 offset = {gain.x * 255.0f, gain.y * 255.0f, gain.z * 255.0f};

  constexpr auto clamp8 = [] __device__(const float v) {
    return static_cast<unsigned char>(fminf(fmaxf(v, 0.0f), 255.0f));
  };

  color out_col;
  out_col.r = clamp8(static_cast<float>(in_col.r) + offset.x * multiplier);
  out_col.g = clamp8(static_cast<float>(in_col.g) + offset.y * multiplier);
  out_col.b = clamp8(static_cast<float>(in_col.b) + offset.z * multiplier);
  out_col.a = in_col.a;

  return thrust::make_tuple(thrust::get<0>(point), out_col,
                            thrust::get<2>(point));
}

} // namespace pc::operators::cuda
