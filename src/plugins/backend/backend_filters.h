#pragma once

#include <cmath>
#include <config/transform_config.h>
#include <pointcaster/core_types.h>

#ifdef __CUDACC__
#define PC_DEVICE_FUNC __host__ __device__
#include <thrust/tuple.h>
#else
#define PC_DEVICE_FUNC
#endif

namespace pc::backend::filter {

static constexpr position invalid_position_value{-32768, -32768, 32767};

struct TransformFilterParameters {
  float position_x, position_y, position_z;
  float rotation_matrix[9];
  float scale_x, scale_y, scale_z;
  float input_translation_x, input_translation_y, input_translation_z;
  float min_x, min_y, min_z;
  float max_x, max_y, max_z;
  int sample;

  static inline TransformFilterParameters
  from_config(const pc::TransformConfiguration &transform_config) {

    const auto &position = transform_config.position.value();
    const auto &rotation = transform_config.rotation.value();
    const auto &scale = transform_config.scale.value();
    const auto &input_translation = transform_config.input_translation.value();
    const auto &min_bound = transform_config.min_bound.value();
    const auto &max_bound = transform_config.max_bound.value();

    constexpr float deg_2_rad = 3.14159265358979f / 180.0f;
    const float cx = std::cos(rotation.x * deg_2_rad);
    const float sxr = std::sin(rotation.x * deg_2_rad);
    const float cy = std::cos(rotation.y * deg_2_rad);
    const float sy = std::sin(rotation.y * deg_2_rad);
    const float cz = std::cos(rotation.z * deg_2_rad);
    const float szr = std::sin(rotation.z * deg_2_rad);

    return {.position_x = position.x * 1000.f,
            .position_y = position.y * 1000.f,
            .position_z = position.z * 1000.f,
            .rotation_matrix = {cy * cz + sy * sxr * szr,
                                sy * sxr * cz - cy * szr, sy * cx, cx * szr,
                                cx * cz, -sxr, cy * sxr * szr - sy * cz,
                                sy * szr + cy * sxr * cz, cy * cx},
            .scale_x = scale.x,
            .scale_y = scale.y,
            .scale_z = scale.z,
            .input_translation_x = input_translation.x * 1000.f,
            .input_translation_y = input_translation.y * 1000.f,
            .input_translation_z = input_translation.z * 1000.f,
            .min_x = min_bound.x * 1000.f,
            .min_y = min_bound.y * 1000.f,
            .min_z = min_bound.z * 1000.f,
            .max_x = max_bound.x * 1000.f,
            .max_y = max_bound.y * 1000.f,
            .max_z = max_bound.z * 1000.f,
            .sample = transform_config.sample.value()};
  }
};

PC_DEVICE_FUNC inline position
transform(position pos, const TransformFilterParameters &param) {
  float x = pos.x + param.input_translation_x;
  float y = pos.y + param.input_translation_y;
  float z = pos.z + param.input_translation_z;

  x *= param.scale_x;
  y *= param.scale_y;
  z *= param.scale_z;

  const auto &r = param.rotation_matrix;
  const float rx = r[0] * x + r[1] * y + r[2] * z;
  const float ry = r[3] * x + r[4] * y + r[5] * z;
  const float rz = r[6] * x + r[7] * y + r[8] * z;

  return {static_cast<int16_t>(rx + param.position_x),
          static_cast<int16_t>(ry + param.position_y),
          static_cast<int16_t>(rz + param.position_z)};
}

PC_DEVICE_FUNC inline bool in_bounds(position pos,
                                     const TransformFilterParameters &param) {
  return (pos.x >= param.min_x && pos.x <= param.max_x) &&
         (pos.y >= param.min_y && pos.y <= param.max_y) &&
         (pos.z >= param.min_z && pos.z <= param.max_z);
}

PC_DEVICE_FUNC inline bool sample(uint32_t idx,
                                  const TransformFilterParameters &p) {
  return (idx % p.sample) == 0;
}

PC_DEVICE_FUNC inline bool is_valid(position p) {
  return p.x != invalid_position_value.x || p.y != invalid_position_value.y ||
         p.z != invalid_position_value.z;
}

PC_DEVICE_FUNC inline position_bounds as_bounds(const position &p) {
  return {p, p};
}

PC_DEVICE_FUNC inline position_bounds merge_bounds(const position_bounds &a,
                                                   const position_bounds &b) {
  return {{a.min.x < b.min.x ? a.min.x : b.min.x,
           a.min.y < b.min.y ? a.min.y : b.min.y,
           a.min.z < b.min.z ? a.min.z : b.min.z},
          {a.max.x < b.max.x ? b.max.x : a.max.x,
           a.max.y < b.max.y ? b.max.y : a.max.y,
           a.max.z < b.max.z ? b.max.z : a.max.z}};
}

} // namespace pc::backend::filter