#pragma once

#include <config/output_value.h>
#include <plugins/backend/backend_types.h>
#include <pointcaster/point_cloud.h>
#include <rfl/DefaultVal.hpp>
#include <rfl/Literal.hpp>
#include <string>

namespace pc::operators {

class ClusterExtractionOperator;

struct ClusterExtractionConfiguration {
  std::string id;                      // @hidden
  rfl::DefaultVal<std::string> label;  // @hidden
  rfl::DefaultVal<bool> active = true; // @hidden

  // voxelisation
  rfl::DefaultVal<int> voxel_leaf_size = 200;        // @minmax(100, 1000)
  rfl::DefaultVal<int> minimum_points_per_voxel = 0; // @minmax(0, 1000)

  rfl::DefaultVal<bool> filter_outlier_voxels = false;
  rfl::DefaultVal<int> outlier_filter_voxel_count = 30; // @minmax(1, 200)
  rfl::DefaultVal<float> outlier_filter_deviation_threshold =
      1.0f; // @minmax(0, 10)

  // clustering
  rfl::DefaultVal<int> cluster_tolerance = 270;       // @minmax(120, 1200)
  rfl::DefaultVal<int> cluster_voxel_count_min = 10;  // @minmax(3, 1000)
  rfl::DefaultVal<int> cluster_voxel_count_max = 100; // @minmax(3, 1000)

  rfl::DefaultVal<int> cluster_match_tolerance = 500; // @minmax(0, 5000)
  rfl::DefaultVal<int> cluster_timeout_ms = 100;      // @minmax(0, 5000)

  // output fields
  Output<VoxelisedCloudPtr> voxels;
  Output<int> voxel_count = 0;
  Output<AabbListPtr> clusters;
  Output<PointCloudPtr> cluster_centroids;
  Output<int> cluster_count = 0;

  rfl::DefaultVal<BackendType> backend = BackendType::CPU; // @hidden

  using OperatorType = ClusterExtractionOperator;
  using Tag = rfl::Literal<"clusterExtraction">;
  static constexpr auto PluginName = "ClusterExtractionOperator";
};

} // namespace pc::operators
