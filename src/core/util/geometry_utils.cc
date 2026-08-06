#include "geometry_utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace pc {

constexpr float deg_to_rad = 3.14159265358979323846f / 180.0f;

float4x4 multiply(const float4x4 &a, const float4x4 &b) {
  float4x4 out;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      float sum = 0.0f;
      for (int k = 0; k < 4; ++k)
        sum += a.values[size_t(r * 4 + k)] * b.values[size_t(k * 4 + c)];
      out.values[size_t(r * 4 + c)] = sum;
    }
  return out;
}

float4x4 translation_matrix(const float3 &t) {
  float4x4 m;
  m.values[3] = t.x;
  m.values[7] = t.y;
  m.values[11] = t.z;
  return m;
}

float4x4 scale_matrix(const float3 &s) {
  float4x4 m;
  m.values[0] = s.x;
  m.values[5] = s.y;
  m.values[10] = s.z;
  return m;
}

float4x4 rotation_matrix(const float3 &euler_degrees) {
  const float rx = euler_degrees.x * deg_to_rad;
  const float ry = euler_degrees.y * deg_to_rad;
  const float rz = euler_degrees.z * deg_to_rad;

  const float cx = std::cos(rx), sx = std::sin(rx);
  const float cy = std::cos(ry), sy = std::sin(ry);
  const float cz = std::cos(rz), sz = std::sin(rz);

  float4x4 mat_x;
  mat_x.values[5] = cx;
  mat_x.values[6] = -sx;
  mat_x.values[9] = sx;
  mat_x.values[10] = cx;

  float4x4 mat_y;
  mat_y.values[0] = cy;
  mat_y.values[2] = sy;
  mat_y.values[8] = -sy;
  mat_y.values[10] = cy;

  float4x4 mat_z;
  mat_z.values[0] = cz;
  mat_z.values[1] = -sz;
  mat_z.values[4] = sz;
  mat_z.values[5] = cz;

  return multiply(mat_y, multiply(mat_x, mat_z));
}

decomposed_transform decompose_transform(const pc::float4x4 &matrix) {
  using RowMajorMatrix4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

  const Eigen::Map<const RowMajorMatrix4f> eigen_transform(
      matrix.values.data());

  const auto translation = eigen_transform.block<3, 1>(0, 3);

  const Eigen::Matrix3f rotation_matrix = eigen_transform.block<3, 3>(0, 0);

  Eigen::Quaternionf eigen_rotation(rotation_matrix);
  eigen_rotation.normalize();

  return decomposed_transform{
      .position =
          pc::float3{
              translation.x(),
              translation.y(),
              translation.z(),
          },
      .rotation =
          pc::quaternion{
              .scalar = eigen_rotation.w(),
              .x = eigen_rotation.x(),
              .y = eigen_rotation.y(),
              .z = eigen_rotation.z(),
          },
  };
}

} // namespace pc