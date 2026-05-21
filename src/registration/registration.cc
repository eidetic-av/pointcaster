#include "registration.h"

#include <Eigen/Core>
#include <conversion/eigen.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp6d.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <span>
#include <stdexcept>

namespace pc::registration {

namespace {

auto to_pcl_cloud(const PointCloud &cloud) {
  auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();
  pcl_cloud->reserve(cloud.size());
  for (std::size_t i = 0; i < cloud.size(); ++i) {
    const auto &pos = cloud.positions[i];
    const auto &col = cloud.colors[i];
    pcl::PointXYZRGBA pt;
    pt.x = static_cast<float>(pos.x);
    pt.y = static_cast<float>(pos.y);
    pt.z = static_cast<float>(pos.z);
    pt.r = col.r;
    pt.g = col.g;
    pt.b = col.b;
    pt.a = col.a;
    pcl_cloud->push_back(pt);
  }
  return pcl_cloud;
}

auto downsample(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,
                float leaf_size) {
  if (leaf_size <= 0.0f) return cloud;
  auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>();
  pcl::VoxelGrid<pcl::PointXYZRGBA> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(leaf_size, leaf_size, leaf_size);
  vg.filter(*filtered);
  return filtered;
}

} // namespace

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

RefinementResult refine_alignment(const PointCloud &source,
                                  const PointCloud &target,
                                  const pc::float4x4 &initial_guess,
                                  const RefinementParams &params) {
  auto src = downsample(to_pcl_cloud(source), params.voxel_leaf_size);
  auto tgt = downsample(to_pcl_cloud(target), params.voxel_leaf_size);

  const Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> guess(
      initial_guess.values.data());

  pcl::GeneralizedIterativeClosestPoint6D icp;
  icp.setInputSource(src);
  icp.setInputTarget(tgt);
  icp.setMaxCorrespondenceDistance(params.max_correspondence_distance);
  icp.setMaximumIterations(params.max_iterations);
  icp.setTransformationEpsilon(params.transformation_epsilon);
  icp.setEuclideanFitnessEpsilon(params.fitness_epsilon);

  pcl::PointCloud<pcl::PointXYZRGBA> aligned;
  icp.align(aligned, guess.matrix());

  return {
      .transform = to_float4x4(icp.getFinalTransformation()),
      .fitness_score = static_cast<float>(icp.getFitnessScore()),
      .converged = icp.hasConverged(),
  };
}

} // namespace pc::registration