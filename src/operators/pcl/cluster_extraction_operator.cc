#include "../../client_sync/updates.h"
#include "../../logger.h"
#include "../../profiling.h"
#include "../../publisher/publisher.h"
#include "../../session.gen.h"
#include "../../string_map.h"
#include "cluster_extraction_operator.gen.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <oneapi/tbb.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>

namespace pc::operators::pcl_cpu {

ClusterExtractionPipeline &
ClusterExtractionPipeline::instance(const std::string_view session_id) {
  static pc::string_map<ClusterExtractionPipeline> instances;
  auto [it, inserted] = instances.try_emplace(session_id, session_id);
  if (inserted)
    pc::logger->info("Created new cluster extraction pipeline instance");
  return it->second;
}

ClusterExtractionPipeline::ClusterExtractionPipeline(
    const std::string_view host_session_id)
    : session_id(host_session_id) {
  auto max_concurrency = tbb::info::default_concurrency();
  input_queue.set_capacity(max_concurrency);

  pc::logger->info(
      "Starting Cluster Extraction TBB Pipeline (max_concurrency = {})",
      max_concurrency);

  _host_thread = std::jthread([this, max_concurrency](std::stop_token st) {
    if (st.stop_requested()) return;
    tbb::parallel_pipeline(
        max_concurrency,
        tbb::make_filter<void, InputFrame *>(tbb::filter_mode::serial_in_order,
                                             IngestTask{input_queue, st}) &
            tbb::make_filter<InputFrame *, void>(
                tbb::filter_mode::parallel,
                ExtractTask{session_id, current_voxels, current_clusters,
                            current_cluster_pca}));
  });
}

ClusterExtractionPipeline::~ClusterExtractionPipeline() {
  _host_thread.request_stop();
}

// TODO swap raw ptrs for shared pointers in pipeline stages

ClusterExtractionPipeline::InputFrame *
ClusterExtractionPipeline::IngestTask::operator()(tbb::flow_control &fc) const {
  using namespace std::chrono_literals;
  auto *new_frame = new InputFrame;
  while (!input_queue.try_pop(*new_frame)) {
    if (st.stop_requested()) {
      delete new_frame;
      fc.stop();
      return nullptr;
    }
    // TODO this sleep time is arbitrary, should probs be more thought put in?
    std::this_thread::sleep_for(2ms);
  }
  return new_frame;
}

void ClusterExtractionPipeline::ExtractTask::operator()(
    ClusterExtractionPipeline::InputFrame *frame) const {
  if (frame == nullptr) return;
  using namespace std::chrono;
  using namespace std::chrono_literals;
  using namespace pc::profiling;

  ProfilingZone function_zone("ClusterExtractionPipeline::ExtractTask::()");

  auto start_time = system_clock::now();

  auto &config = frame->extraction_config;
  const auto operator_name = operator_friendly_names.at(config.id);
  const auto session_label = session_label_from_id[session_id];

  // TODO check timestamp against current time
  // in order to throw away older frames

  // TODO configuration options for voxel sampling and clustering
  {}

  auto &positions = frame->positions;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->points.resize(positions.size());
  {
    ProfilingZone conversion_zone("Convert positions to PCL");
    thrust::transform(positions.begin(), positions.end(), cloud->points.begin(),
                      PositionToPointXYZ());
  }

  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  pcl::PointCloud<pcl::PointXYZ>::Ptr voxelised_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  if (!cloud->empty()) {
    ProfilingZone voxel_zone("Voxel grid filter");
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(config.voxel_leaf_size, config.voxel_leaf_size,
                           config.voxel_leaf_size); // mm
    voxel_grid.setMinimumPointsNumberPerVoxel(config.minimum_points_per_voxel);
    voxel_grid.filter(*voxelised_cloud);
  }

  if (config.filter_outlier_voxels && !voxelised_cloud->empty()) {
    ProfilingZone voxel_zone("Outlier filter");
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> outlier_filter;
    outlier_filter.setInputCloud(voxelised_cloud);
    auto mean_k = config.outlier_filter_voxel_count;
    if (mean_k <= 0) mean_k = 1;
    outlier_filter.setMeanK(mean_k);
    auto std_deviation_threshold = config.outlier_filter_deviation_threshold;
    if (std_deviation_threshold <= 0.0f) std_deviation_threshold = 0.0f;
    outlier_filter.setStddevMulThresh(std_deviation_threshold);
    outlier_filter.filter(*voxelised_cloud);
  }

  current_voxels.store(voxelised_cloud);

  std::vector<Cluster> updated_clusters;
  std::vector<pcl::PointIndices> cluster_indices;

  if (!voxelised_cloud->empty()) {
    ProfilingZone clustering_zone("Clustering");
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
    {
      ProfilingZone create_kdtree_zone("Create KDTree");
      tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(
          new pcl::search::KdTree<pcl::PointXYZ>);
      tree->setInputCloud(voxelised_cloud);
    }

    {
      ProfilingZone extract_clusters_zone("Extract clusters");
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance(config.cluster_tolerance); // mm
      ec.setMinClusterSize(config.cluster_voxel_count_min);
      ec.setMaxClusterSize(config.cluster_voxel_count_max);
      ec.setSearchMethod(tree);
      ec.setInputCloud(voxelised_cloud);
      ec.extract(cluster_indices);
    }

    tbb::concurrent_vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
        cluster_point_clouds;
    tbb::concurrent_vector<pc::AABB> cluster_bounds;
    {
      ProfilingZone gen_clustered_point_clouds_zone(
          "Seperate clustered point clouds");

      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, cluster_indices.size()),
          [&cluster_indices, &voxelised_cloud, &cluster_point_clouds,
           &cluster_bounds](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
              const auto &indices = cluster_indices[i];

              pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
                  new pcl::PointCloud<pcl::PointXYZ>);
              for (const auto &index : indices.indices) {
                cloud_cluster->points.push_back(voxelised_cloud->points[index]);
              }
              cloud_cluster->width = cloud_cluster->points.size();
              cloud_cluster->height = 1;
              cloud_cluster->is_dense = true;

              // Compute the bounding box
              pcl::PointXYZ min_pt, max_pt;
              pcl::getMinMax3D(*cloud_cluster, min_pt, max_pt);

              pc::types::Float3 min_pt_f(min_pt.x, min_pt.y, min_pt.z);
              pc::types::Float3 max_pt_f(max_pt.x, max_pt.y, max_pt.z);

              cluster_point_clouds.push_back(cloud_cluster);
              cluster_bounds.push_back(pc::AABB(min_pt_f, max_pt_f));
            }
          });
    }

    {
      ProfilingZone match_zone("Matching clusters");
      std::unordered_set<const Cluster *> matched_existing_clusters;
      auto existing_clusters = current_clusters.load();
      if (!existing_clusters) {
        existing_clusters = std::make_shared<std::vector<Cluster>>();
      }
      auto now = system_clock::now();

      for (int i = 0; i < cluster_bounds.size(); ++i) {
        const auto &new_bound = cluster_bounds[i];
        const auto &new_point_cloud = cluster_point_clouds[i];

        static constexpr auto matchClusters =
            [](const pc::AABB &a, const pc::AABB &b, float tolerance) {
              auto centroid_a = a.center();
              auto centroid_b = b.center();
              auto dx = centroid_a.x - centroid_b.x;
              auto dy = centroid_a.y - centroid_b.y;
              auto dz = centroid_a.z - centroid_b.z;
              auto distance = std::sqrt(dx * dx + dy * dy + dz * dz);
              return distance <= tolerance;
            };

        bool matched = false;
        for (auto &existing_cluster : *existing_clusters) {
          if (matchClusters(existing_cluster.bounding_box, new_bound,
                            config.cluster_match_tolerance)) {
            // found a match to an existing cluster
            existing_cluster.bounding_box = new_bound;
            existing_cluster.last_seen_time = now;
            existing_cluster.point_cloud = new_point_cloud;
            updated_clusters.push_back(existing_cluster);
            matched_existing_clusters.insert(&existing_cluster);
            matched = true;
            break;
          }
        }
        if (!matched) {
          // new cluster with no prior match
          updated_clusters.push_back(Cluster{new_bound, now, new_point_cloud});
        }
      }

      // add existing clusters that were not matched but haven't expired
      for (const auto &existing_cluster : *existing_clusters) {
        if (matched_existing_clusters.find(&existing_cluster) ==
            matched_existing_clusters.end()) {
          // this existing cluster has not been matched, check its timeout
          if (now - existing_cluster.last_seen_time <=
              milliseconds(config.cluster_timeout_ms)) {
            // and add it to our updated list if it should still be alive
            updated_clusters.push_back(existing_cluster);
          }
        }
      }
    }

    if (config.publish_clusters) {
      ProfilingZone publish_zone("Publish clusters");
      // the following transforms our clusters gives us their bounding
      // boxes in std containers, which are easily publishable
      using std_aabb = std::array<std::array<float, 3>, 2>;
      std::vector<std_aabb> aabbs(updated_clusters.size());
      std::ranges::transform(
          updated_clusters, aabbs.begin(), [](const Cluster &cluster) {
            const auto &aabb = cluster.bounding_box;
            return std_aabb({{aabb.min.x, aabb.min.y, aabb.min.z},
                             {aabb.max.x, aabb.max.y, aabb.max.z}});
          });
      publisher::publish_all(operator_name, aabbs, {session_label, "clusters"});
    }

    if (config.calculate_pca) {
      ProfilingZone publish_zone("Principal Component Analysis");

      // we use a parallel for to loop through each cluster and use
      // Principal Component Analysis to calculate the principal axis and the
      // cluster's span (width) across the principal axis

      tbb::concurrent_vector<PCAResult> cluster_pca_results;

      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, updated_clusters.size()),
          [&updated_clusters,
           &cluster_pca_results](const tbb::blocked_range<size_t> &r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {

              const auto &cluster = updated_clusters[i];
              const auto &cluster_cloud = cluster.point_cloud;

              if (!cluster_cloud || cluster_cloud->empty()) continue;

              // compute the Principal Component Analysis on the cluster
              Eigen::Matrix3f eigenvectors;
              Eigen::Vector3f eigenvalues;
              try {
                pcl::PCA<pcl::PointXYZ> cluster_pca;
                cluster_pca.setInputCloud(cluster_cloud);
                eigenvectors = cluster_pca.getEigenVectors();
                eigenvalues = cluster_pca.getEigenValues();
              } catch (const std::exception &e) {
                pc::logger->error("PCA Error: {}", e.what());
                continue;
              }

              // we need to identify the vertical (up-down) axis by comparing
              // with our global up-axis.
              // - this is so that we ignore up-down and just get horizontal
              // span
              static const Eigen::Vector3f up_axis(0, 1, 0);
              int up_down_index = 0;
              float max_dot_product =
                  std::abs(eigenvectors.col(0).dot(up_axis));
              for (int i = 1; i < 3; ++i) {
                float dot_product = std::abs(eigenvectors.col(i).dot(up_axis));
                if (dot_product > max_dot_product) {
                  max_dot_product = dot_product;
                  up_down_index = i;
                }
              }
              // then identify the horizontal axis with the largest eigenvalue
              int horizontal_axis_index = (up_down_index == 0) ? 1 : 0;
              if (eigenvalues(horizontal_axis_index) <
                  eigenvalues(3 - up_down_index - horizontal_axis_index)) {
                horizontal_axis_index =
                    3 - up_down_index - horizontal_axis_index;
              }
              // select the corresponding eigenvector
              Eigen::Vector3f horizontal_axis =
                  eigenvectors.col(horizontal_axis_index);

              // calculate the centroid of the cluster
              // TODO its probably useful to do this elsewhere
              Eigen::Vector4f centroid;
              pcl::compute3DCentroid(*cluster_cloud, centroid);

              // project points onto the selected horizontal axis and find
              // min/max projections
              // - set up the result variables for this:
              static constexpr auto max_f = std::numeric_limits<float>::max();
              static constexpr auto min_f =
                  std::numeric_limits<float>::lowest();
              float min_projection = max_f;
              float max_projection = min_f;
              Eigen::Vector3f min_point, max_point;
              // also find min/max extremes for each axis (unprojected, aabb
              // aligned)
              // - set up these results:
              std::array<PCAResult::minMax, 3> axis_extremes;
              for (int axis = 0; axis < 3; axis++) {
                axis_extremes[axis] = {PCAResult::float3{max_f, max_f, max_f},
                                       PCAResult::float3{min_f, min_f, min_f}};
              }

              for (const auto &point : cluster_cloud->points) {
                Eigen::Vector3f p(point.x, point.y, point.z);
                // projected along the principal horizontal axis
                float projection =
                    (p - centroid.head<3>()).dot(horizontal_axis);
                if (projection < min_projection) {
                  min_projection = projection;
                  min_point = p;
                }
                if (projection > max_projection) {
                  max_projection = projection;
                  max_point = p;
                }
                // unprojected, aabb aligned min/max axis values
                for (int axis = 0; axis < 3; ++axis) {
                  if (p[axis] < axis_extremes[axis][0][axis]) {
                    axis_extremes[axis][0] = {p.x(), p.y(), p.z()};
                  }
                  if (p[axis] > axis_extremes[axis][1][axis]) {
                    axis_extremes[axis][1] = {p.x(), p.y(), p.z()};
                  }
                }
              }
              // and collect the results
              cluster_pca_results.push_back(PCAResult{
                  .centroid = {centroid.x(), centroid.y(), centroid.z()},
                  .principal_axis_span = {std::array<float, 3>{min_point.x(),
                                                               min_point.y(),
                                                               min_point.z()},
                                          std::array<float, 3>{max_point.x(),
                                                               max_point.y(),
                                                               max_point.z()}},
                  .axis_extremes = axis_extremes});
            }
          });

      // move result to host instance
      auto output_pca_results = std::make_shared<std::vector<PCAResult>>(
          std::move_iterator(cluster_pca_results.begin()),
          std::move_iterator(cluster_pca_results.end()));

      // // auto output_pca_results =
      // std::make_shared<std::vector<PCAResult>>(principal_spans.size());
      // // std::copy(cluster_pca_results.begin(), cluster_pca_results.end(),
      // output_pca_results->begin());
      // // current_cluster_pca.store(output_pca_results);

      if (config.publish_pca) {
        ProfilingZone publish_zone("Publish PCA results");
        // publish the structure as a flat array, but also as individual
        // channels
        const auto cluster_count = output_pca_results->size();

        // TODO it would be useful for publisher::publish_all to handle
        // reflecting over types to do both the flattened and specific channel
        // publishing automatically

        // using float3 = std::array<float, 3>;
        // using minMax = std::array<float3, 2>;
        // std::vector<float3> centroids;
        // centroids.reserve(cluster_count);
        // std::vector<minMax> pca_spans;
        // pca_spans.reserve(cluster_count);
        // std::vector<std::array<minMax, 3>> axis_extremes;
        // axis_extremes.reserve(cluster_count);

        // for (const auto& pca_result : *output_pca_results) {
        // centroids.push_back(pca_result.centroid);
        // pca_spans.push_back(pca_result.principal_axis_span);
        // axis_extremes.push_back(pca_result.axis_extremes);
        // }

        static const auto id_string = "pcl";
        using FlattenedPCAResult =
            decltype(std::declval<PCAResult>().flattened());
        std::vector<FlattenedPCAResult> flattened_pca_results(cluster_count);
        std::transform(output_pca_results->begin(), output_pca_results->end(),
                       flattened_pca_results.begin(),
                       [](const auto &result) { return result.flattened(); });
        publisher::publish_all("flattened", flattened_pca_results,
                               {id_string, "pca"});

        // specific channels
        // publisher::publish_all("centroids", centroids, { id_string, "pca" });
        // publisher::publish_all("pca_spans", pca_spans, { id_string, "pca" });
        // publisher::publish_all("extremes", axis_extremes, { id_string, "pca"
        // });
      }

      current_cluster_pca.store(std::move(output_pca_results));
    }

    // and latest clusters onto main thread now we're done with them
    current_clusters.store(
        std::make_shared<std::vector<Cluster>>(std::move(updated_clusters)));
  }

  if (config.publish_voxels) {
    ProfilingZone publish_voxels_zone("Publish voxels");

    // when publishing voxels, we send the position vector x,y,z with w
    // representing the integer index of the cluster the associated voxel

    // TODO we can probably optimise this by removing the serial loops inside
    // the first parallel_for

    size_t voxel_count = voxelised_cloud->points.size();
    pc::logger->debug("{}.voxel_count: {}", config.id, voxel_count);

    // map each point to its cluster id (default -1 if no cluster associated)
    // returning a flat array that we can parallel iterate over next
    std::vector<int> cluster_ids(voxel_count, -1);
    tbb::parallel_for(static_cast<size_t>(0), cluster_indices.size(),
                      [&](size_t cluster_id) {
                        for (auto idx : cluster_indices[cluster_id].indices) {
                          cluster_ids[idx] = static_cast<int>(cluster_id);
                        }
                      });

    // now we can loop through each entry in parallel and
    // create our voxel list with associated cluster ids
    std::vector<std::array<float, 4>> voxels_with_cluster_id(voxel_count);
    tbb::parallel_for(
        static_cast<size_t>(0), voxel_count, [&](size_t voxel_id) {
          const auto &position = voxelised_cloud->points[voxel_id];
          voxels_with_cluster_id[voxel_id] = {
              position.x, position.y, position.z,
              static_cast<float>(cluster_ids[voxel_id])};
        });

    // only publish an empty voxel array a single time, not every frame
    // so keep track of that inside a map
    static std::unordered_map<uid, bool> has_published_empty_voxels;
    if (!has_published_empty_voxels.contains(config.id)) {
      has_published_empty_voxels[config.id] = false;
    }

    if (voxel_count > 0) {
      const auto &operator_name = operator_friendly_names.at(config.id);
      publisher::publish_all(operator_name, voxels_with_cluster_id,
                             {session_label, "voxels"});
      has_published_empty_voxels[config.id] = false;
    } else if (!has_published_empty_voxels[config.id]) {
      const auto &operator_name = operator_friendly_names.at(config.id);
      publisher::publish_all(operator_name, voxels_with_cluster_id,
                             {session_label, "voxels"});
      has_published_empty_voxels[config.id] = true;
    }
  }

  delete frame;
}
} // namespace pc::operators::pcl_cpu