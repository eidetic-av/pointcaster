#pragma once

#include "../../structs.h"
#include "uniform_gain_operator.gen.h"
#include <thrust/functional.h>

namespace pc::operators::cuda {

using pc::types::color;
using pc::types::position;
using pc::types::Float3;

__device__ indexed_point_t
UniformGainOperator::operator()(indexed_point_t point) const {
  const color in_col = thrust::get<1>(point);
  const float gain    = _config.gain;        // now in [-1,1]
  const float offset  = gain * 255.0f;       // map to [-255,255]

  constexpr auto clamp8 = [] (float v) {
    return static_cast<unsigned char>(fminf(fmaxf(v, 0.0f), 255.0f));
  };

  color out_col;
  out_col.r = clamp8(static_cast<float>(in_col.r) + offset);
  out_col.g = clamp8(static_cast<float>(in_col.g) + offset);
  out_col.b = clamp8(static_cast<float>(in_col.b) + offset);
  out_col.a = in_col.a;

  return thrust::make_tuple(
      thrust::get<0>(point),
      out_col,
      thrust::get<2>(point)
  );
}

} // namespace pc::operators::cuda
