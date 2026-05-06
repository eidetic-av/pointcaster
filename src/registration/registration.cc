#include "registration.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <Eigen/Core>
#include <conversion/eigen.h>

#include <span>
#include <stdexcept>

namespace pc::registration {

pc::float4x4 compute_rigid_transform(std::span<const pc::float3> source,
                                     std::span<const pc::float3> target) {
  if (source.size() != target.size() || source.size() < 3) {
    throw std::invalid_argument("need >= 3 matched pairs of equal size");
  }

  pcl::PointCloud<pcl::PointXYZ> source_cloud;
  pcl::PointCloud<pcl::PointXYZ> target_cloud;

  source_cloud.reserve(source.size());
  target_cloud.reserve(target.size());

  for (std::size_t point_index = 0; point_index < source.size();
       ++point_index) {
    const auto &source_point = source[point_index];
    const auto &target_point = target[point_index];

    source_cloud.emplace_back(source_point.x, source_point.y, source_point.z);
    target_cloud.emplace_back(target_point.x, target_point.y, target_point.z);
  }

  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

  pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>
      svd;

  svd.estimateRigidTransformation(source_cloud, target_cloud, transform);

  return to_float4x4(transform);
}

} // namespace pc::registration