#pragma once

#include <cmath>
#include <config/transform_config.h>

#ifdef __CUDACC__
#define PC_DEVICE_FUNC __host__ __device__
#include <thrust/tuple.h>
#else
#define PC_DEVICE_FUNC
#endif

namespace pc::backend::filter {

struct TransformFilterParameters {
  float scale_x, scale_y, scale_z;
  float rotation_matrix[9];
  float translate_x, translate_y, translate_z;
  float min_x, min_y, min_z;
  float max_x, max_y, max_z;

  static inline TransformFilterParameters
  from_config(const pc::TransformConfiguration &transform_config) {

    const auto &translation = transform_config.position.value();
    const auto &rotation = transform_config.rotation.value();
    const auto &scale = transform_config.scale.value();
    const auto &min_bound = transform_config.min_bound.value();
    const auto &max_bound = transform_config.max_bound.value();

    constexpr float deg_2_rad = 3.14159265358979f / 180.0f;
    const float cx = std::cos(rotation.x * deg_2_rad);
    const float sxr = std::sin(rotation.x * deg_2_rad);
    const float cy = std::cos(rotation.y * deg_2_rad);
    const float sy = std::sin(rotation.y * deg_2_rad);
    const float cz = std::cos(rotation.z * deg_2_rad);
    const float szr = std::sin(rotation.z * deg_2_rad);

    return {.scale_x = scale.x,
            .scale_y = scale.y,
            .scale_z = scale.z,
            .rotation_matrix = {cy * cz, cy * szr, -sy,
                                sxr * sy * cz - cx * szr,
                                sxr * sy * szr + cx * cz, sxr * cy,
                                cx * sy * cz + sxr * szr,
                                cx * sy * szr - sxr * cz, cx * cy},
            .translate_x = translation.x * 1000.f,
            .translate_y = translation.y * 1000.f,
            .translate_z = translation.z * 1000.f,
            .min_x = min_bound.x * 1000.f,
            .min_y = min_bound.y * 1000.f,
            .min_z = min_bound.z * 1000.f,
            .max_x = max_bound.x * 1000.f,
            .max_y = max_bound.y * 1000.f,
            .max_z = max_bound.z * 1000.f};
  }
};

PC_DEVICE_FUNC inline position
transform(position pos, const TransformFilterParameters &param) {
  const float x = pos.x * param.scale_x;
  const float y = pos.y * param.scale_y;
  const float z = pos.z * param.scale_z;

  // TODO
  // rotation matrix values use ZYX, maybe cross check to see if its the
  // rotation order we want for sure
  const auto &r = param.rotation_matrix;
  const float out_x = (r[0] * x + r[1] * y + r[2] * z) + param.translate_x;
  const float out_y = (r[3] * x + r[4] * y + r[5] * z) + param.translate_y;
  const float out_z = (r[6] * x + r[7] * y + r[8] * z) + param.translate_z;

  return {static_cast<int16_t>(out_x), static_cast<int16_t>(out_y),
          static_cast<int16_t>(out_z)};
}

PC_DEVICE_FUNC inline bool in_bounds(position pos,
                                     const TransformFilterParameters &param) {
  return (pos.x >= param.min_x && pos.x <= param.max_x) &&
         (pos.y >= param.min_y && pos.y <= param.max_y) &&
         (pos.z >= param.min_z && pos.z <= param.max_z);
}

} // namespace pc::backend::filter