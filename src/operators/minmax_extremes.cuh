#include "../structs.h"
#include "operator.h"
#include <cstdint>

namespace pc::operators {

using position = pc::types::position;

struct Extremes {
  position min_x_pos;
  position max_x_pos;
  position min_y_pos;
  position max_y_pos;
  position min_z_pos;
  position max_z_pos;
};

// Initialise with sentinels
__host__ __device__ Extremes make_initial_extremes() {
  constexpr auto plus_inf = std::numeric_limits<int16_t>::max();
  constexpr auto minus_inf = -std::numeric_limits<int16_t>::max();
  return {.min_x_pos = {plus_inf, 0, 0},
          .max_x_pos = {minus_inf, 0, 0},
          .min_y_pos = {0, plus_inf, 0},
          .max_y_pos = {0, minus_inf, 0},
          .min_z_pos = {0, 0, plus_inf},
          .max_z_pos = {0, 0, minus_inf}};
}

struct AsExtremes {
  __host__ __device__ Extremes operator()(const indexed_point_t &t) const {
    position p = thrust::get<0>(t);
    Extremes e{};
    e.min_x_pos = p;
    e.max_x_pos = p;
    e.min_y_pos = p;
    e.max_y_pos = p;
    e.min_z_pos = p;
    e.max_z_pos = p;
    return e;
  }
};

struct MergeExtremes {
  __host__ __device__ Extremes operator()(const Extremes &a,
                                          const Extremes &b) const {
    Extremes r = a;

    if (b.min_x_pos.x < r.min_x_pos.x) r.min_x_pos = b.min_x_pos;
    if (b.max_x_pos.x > r.max_x_pos.x) r.max_x_pos = b.max_x_pos;

    if (b.min_y_pos.y < r.min_y_pos.y) r.min_y_pos = b.min_y_pos;
    if (b.max_y_pos.y > r.max_y_pos.y) r.max_y_pos = b.max_y_pos;

    if (b.min_z_pos.z < r.min_z_pos.z) r.min_z_pos = b.min_z_pos;
    if (b.max_z_pos.z > r.max_z_pos.z) r.max_z_pos = b.max_z_pos;

    return r;
  }
};

} // namespace pc::operators