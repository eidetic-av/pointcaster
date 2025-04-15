#pragma once

#include "../../aabb.h"
#include "../../structs.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/parallel_pipeline.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/host_vector.h>
#include <vector>

namespace pc::operators::pcl_cpu {

using uid = unsigned long;

struct ClusterExtractionConfiguration {
  uid id;
  bool enabled = true;
  bool draw_voxels = false;
  bool draw_clusters = true;
  int voxel_leaf_size = 200; // @minmax(100, 1000)
  int minimum_points_per_voxel = 0; // @minmax(0, 100)
  bool filter_outlier_voxels = false;
  int outlier_filter_voxel_count = 30;
  float outlier_filter_deviation_threshold = 1.0f;
  bool publish_voxels = false;
  int cluster_tolerance = 270;       // @minmax(120, 1200)
  int cluster_voxel_count_min = 10;  // @minmax(3, 1000)
  int cluster_voxel_count_max = 100; // @minmax(3, 1000)
  int cluster_size_min = 10;         // @minmax(0, 5000)
  int cluster_size_max = 1000;       // @minmax(0, 5000)
  int cluster_timeout_ms = 100;
  int cluster_match_tolerance = 50;
  bool publish_clusters = false;
  bool calculate_pca = true;
  bool draw_pca = true;
  bool publish_pca = false;
};

class ClusterExtractionPipeline {

public:
  struct InputFrame {
    unsigned long timestamp;
    ClusterExtractionConfiguration extraction_config;
    thrust::host_vector<pc::types::position> positions;
  };

  struct PCAResult {
    using float3 = std::array<float, 3>;
    using minMax = std::array<float3, 2>;
    float3 centroid;
    minMax principal_axis_span;
    std::array<minMax, 3> axis_extremes;

    std::array<float,
               (3) +            // centroid
                   (3 * 2) +    // principal axis span
                   (3 * 2 * 3)> // axis_extremes
    flattened() const {
      return {centroid[0],
              centroid[1],
              centroid[2],
              principal_axis_span[0][0],
              principal_axis_span[0][1],
              principal_axis_span[0][2],
              principal_axis_span[1][0],
              principal_axis_span[1][1],
              principal_axis_span[1][2],
              axis_extremes[0][0][0],
              axis_extremes[0][0][1],
              axis_extremes[0][0][2],
              axis_extremes[0][1][0],
              axis_extremes[0][1][1],
              axis_extremes[0][1][2],
              axis_extremes[1][0][0],
              axis_extremes[1][0][1],
              axis_extremes[1][0][2],
              axis_extremes[1][1][0],
              axis_extremes[1][1][1],
              axis_extremes[1][1][2],
              axis_extremes[2][0][0],
              axis_extremes[2][0][1],
              axis_extremes[2][0][2],
              axis_extremes[2][1][0],
              axis_extremes[2][1][1],
              axis_extremes[2][1][2]};
    }
  };

  struct Cluster {
    pc::AABB bounding_box;
    std::chrono::system_clock::time_point last_seen_time;
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud;
  };

  static ClusterExtractionPipeline &instance();

  oneapi::tbb::concurrent_bounded_queue<InputFrame> input_queue;
  std::atomic<pcl::PointCloud<pcl::PointXYZ>::Ptr> current_voxels;
  std::atomic<std::shared_ptr<std::vector<Cluster>>> current_clusters;
  std::atomic<std::shared_ptr<std::vector<PCAResult>>> current_cluster_pca;

  ClusterExtractionPipeline();
  ~ClusterExtractionPipeline();

  ClusterExtractionPipeline(const ClusterExtractionPipeline &) = delete;
  ClusterExtractionPipeline &
  operator=(const ClusterExtractionPipeline &) = delete;
  ClusterExtractionPipeline(ClusterExtractionPipeline &&) = delete;
  ClusterExtractionPipeline &operator=(ClusterExtractionPipeline &&) = delete;

private:
  struct IngestTask {
    oneapi::tbb::concurrent_bounded_queue<InputFrame> &input_queue;
    std::stop_token st;
    InputFrame *operator()(tbb::flow_control &fc) const;
  };

  struct ExtractTask {
    std::atomic<pcl::PointCloud<pcl::PointXYZ>::Ptr> &current_voxels;
    std::atomic<std::shared_ptr<std::vector<Cluster>>> &current_clusters;
    std::atomic<std::shared_ptr<std::vector<PCAResult>>> &current_cluster_pca;
    void operator()(InputFrame *frame) const;
  };
  std::jthread _host_thread;
};

// Util for converting to pcl types
struct PositionToPointXYZ {
  __host__ __device__ pcl::PointXYZ
  operator()(const pc::types::position &pos) const {
    pcl::PointXYZ point;
    point.x = static_cast<float>(pos.x);
    point.y = static_cast<float>(pos.y);
    point.z = static_cast<float>(pos.z);
    return point;
  }
};

// Util for converting from pcl types
struct PointXYZToPosition {
  __host__ __device__ pc::types::position
  operator()(const pcl::PointXYZ &pos) const {
    return pc::types::position{static_cast<short>(pos.x),
                               static_cast<short>(pos.y),
                               static_cast<short>(pos.z)};
  }
};

} // namespace pc::operators::pcl_cpu
