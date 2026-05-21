#include "geometry_utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pc {

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