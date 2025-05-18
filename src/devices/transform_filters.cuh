#pragma once

#include "../structs.h"
#include "device_config.gen.h"
#include <Eigen/Eigen>
#include <libobsensor/ObSensor.hpp>
#include <thrust/functional.h>
#include <thrust/tuple.h>

namespace pc::devices {

using pc::types::color;
using pc::types::position;

typedef thrust::tuple<OBColorPoint, int> ob_point_in_t;
typedef thrust::tuple<position, color, int> indexed_point_t;

__host__ __device__ static inline float as_rad(float deg) {
  constexpr auto mult = 3.141592654f / 180.0f;
  return deg * mult;
};

struct input_transform_filter {
  DeviceTransformConfiguration config;

  __device__ bool check_empty(position pos, color col) const {
    if (pos.x == 0.0f && pos.y == 0.0f && pos.z == 0.0f) return false;
    if (col.r == 0.0f && col.g == 0.0f && col.b == 0.0f) return false;
    return true;
  }

  __device__ bool check_crop(position pos) const {
    auto x = config.flip_x ? -pos.x : pos.x;
    auto y = config.flip_y ? pos.y : -pos.y;
    auto z = config.flip_z ? -pos.z : pos.z;
    return x >= config.crop_x.min && x <= config.crop_x.max &&
           y >= config.crop_y.min && y <= config.crop_y.max &&
           z >= config.crop_z.min && z <= config.crop_z.max;
  }

  __device__ bool sample(int index) const {
    return (index % config.sample) == 0;
  }

  __device__ bool operator()(indexed_point_t point) const {
    auto index = thrust::get<2>(point);
    if (!sample(index)) return false;
    auto pos = thrust::get<0>(point);
    auto col = thrust::get<1>(point);
    if (!check_empty(pos, col)) return false;
    if (!check_crop(pos)) return false;
    return true;
  }
};

struct output_transform_filter {
  DeviceTransformConfiguration config;

  __device__ bool check_bounds(position value) const {

    auto x = config.flip_x ? -value.x : value.x;
    auto y = config.flip_y ? -value.y : value.y;
    auto z = config.flip_z ? -value.z : value.z;

    return x >= config.bound_x.min && x <= config.bound_x.max &&
           y >= config.bound_y.min && y <= config.bound_y.max &&
           z >= config.bound_z.min && z <= config.bound_z.max;
  }

  __device__ bool operator()(indexed_point_t point) const {
    auto pos = thrust::get<0>(point);
    return check_bounds(pos);
  }
};

struct device_transform_filter
    : public thrust::unary_function<indexed_point_t, indexed_point_t> {

  DeviceTransformConfiguration config;

  explicit device_transform_filter(
      DeviceTransformConfiguration &transform_config)
      : config(transform_config) {}

  __device__ indexed_point_t operator()(indexed_point_t point) const {
    using namespace Eigen;

    // unpack input point
    position in_p = thrust::get<0>(point);
    Vector3f pos_vec{static_cast<float>(in_p.x), static_cast<float>(in_p.y),
                     static_cast<float>(in_p.z)};

    // world-space axis flips
    Vector3f axis_flip{config.flip_x ? -1.0f : 1.0f,
                       config.flip_y ? -1.0f : 1.0f,
                       config.flip_z ? -1.0f : 1.0f};
    pos_vec = {pos_vec.x(), -pos_vec.y(), -pos_vec.z()};

    // pre-rotation translate
    pos_vec += Vector3f{config.translate.x * axis_flip.x(),
                        config.translate.y * axis_flip.y(),
                        config.translate.z * axis_flip.z()};

    // flip axes before rotation
    pos_vec = {pos_vec.x() * axis_flip.x(), pos_vec.y() * axis_flip.y(),
               pos_vec.z() * axis_flip.z()};

    // apply quaternion rotation (config.rotation = {x,y,z,w})
    Quaternionf rot_quat{config.rotation.w, config.rotation.x,
                         config.rotation.y, config.rotation.z};
    pos_vec = rot_quat * pos_vec;

    // alignment offset and uniform scale
    pos_vec += Vector3f{config.offset.x, config.offset.y, config.offset.z};
    pos_vec *= config.scale;

    // pack back into 16-bit output
    position out_p = {(short)__float2int_rd(pos_vec.x()),
                      (short)__float2int_rd(pos_vec.y()),
                      (short)__float2int_rd(pos_vec.z()), 0};

    return thrust::make_tuple(out_p, thrust::get<1>(point),
                              thrust::get<2>(point));
  }
};

} // namespace pc::devices