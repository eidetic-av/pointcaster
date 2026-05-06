#pragma once

#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <span>

namespace pc::registration {

// Compute rigid transform that maps source points onto target points.
// Minimum 3 pairs.
pc::float4x4 compute_rigid_transform(std::span<const pc::float3> source,
                                     std::span<const pc::float3> target);

struct RefinementParams {
  float max_correspondence_distance = 50.0f;
  int max_iterations = 50;
  float voxel_leaf_size = 5.0f; // 0 = no downsampling
  double transformation_epsilon = 1e-6;
  double fitness_epsilon = 1e-6;
};

struct RefinementResult {
  pc::float4x4 transform;
  float fitness_score;
  bool converged;
};

RefinementResult refine_alignment(const PointCloud &source,
                                  const PointCloud &target,
                                  const pc::float4x4 &initial_guess,
                                  const RefinementParams &params = {});

} // namespace pc::registration