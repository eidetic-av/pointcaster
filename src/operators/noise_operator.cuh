#include "../structs.h"
#include "noise_operator.h"
#include <cuda_noise.cuh>
#include <thrust/functional.h>

namespace pc::operators {

using pc::types::color;
using pc::types::position;

__device__ indexed_point_t NoiseOperator::operator()(indexed_point_t point) const {

  position pos = thrust::get<0>(point);
  color col = thrust::get<1>(point);
  int index = thrust::get<2>(point);

  float3 pos_f = {(float)pos.x, (float)pos.y, (float)pos.z};

  auto& scale = _config.scale;
  auto& magnitude = _config.magnitude;
  auto& seed = _config.seed;
  auto& repeat = _config.repeat;
  auto& lacunarity = _config.lacunarity;
  auto& decay = _config.decay;

  pos.x += cudaNoise::repeaterSimplex({pos_f.x, pos_f.y, pos_f.z}, scale, seed,
                                     repeat, lacunarity, decay) *
           magnitude;
  pos.y += cudaNoise::repeaterSimplex({pos_f.z, pos_f.x, pos_f.y}, scale, seed,
                                     repeat, lacunarity, decay) *
           magnitude;
  pos.z += cudaNoise::repeaterSimplex({pos_f.y, pos_f.z, pos_f.x}, scale, seed,
                                     repeat, lacunarity, decay) *
           magnitude;

  return thrust::make_tuple(pos, col, index);
};

} // namespace pc::operators
