#include "floor_plane_alignment_operator.gen.h"

#include "../../logger/logger.h"
#include "../../profiling.h"

#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <thread>
#include <thrust/host_vector.h>

namespace pc::operators::pcl_cpu {

FloorPlaneAlignmentOperator::FloorPlaneAlignmentOperator(
    FloorPlaneAlignmentOperatorConfiguration &config)
    : _config(config) {}

FloorPlaneAlignmentOperator::~FloorPlaneAlignmentOperator() {
  if (_alignment_thread.joinable()) { _alignment_thread.request_stop(); }
}

void FloorPlaneAlignmentOperator::update(operator_in_out_t begin,
                                         operator_in_out_t end) {
  std::scoped_lock lock(_mutex);
  if (!_config.enabled) { return; }

  if (_alignment_thread_finished.load()) {
    _alignment_thread.join();
    _alignment_thread = std::jthread{};
    _alignment_thread_finished.store(false);
  }

  // already running
  if (_alignment_thread.joinable()) return;

  // copy positions to host
  auto device_positions_begin = thrust::get<0>(begin.get_iterator_tuple());
  auto device_positions_end = thrust::get<0>(end.get_iterator_tuple());
  thrust::host_vector<pc::types::position> host_positions(
      device_positions_begin, device_positions_end);

  // start one-off thread
  _alignment_thread =
      std::jthread([this, positions = std::move(host_positions)](
                       std::stop_token st) mutable {
        using namespace pc::profiling;
        ProfilingZone zone("FloorPlaneAlignmentOperator::alignment_thread");

        if (st.stop_requested()) return;

        pc::types::Float3 euler_deg{0.f, 0.f, 0.f};
        const bool success = compute_floor_euler_degrees(positions, euler_deg);

        if (st.stop_requested()) return;

        {
          std::scoped_lock lock(_mutex);
          if (success) {
            _config.euler_angles = euler_deg;
          } else {
            pc::logger()->error("Failed to fit to plane!");
          }
          _config.enabled = false;
        }

        _alignment_thread_finished.store(true);
      });
}

static inline float rad_to_deg(float r) {
  return r * (180.0f / 3.14159265358979323846f);
}

static inline pc::types::Float3
quaternion_to_euler_deg_RzRyRx(const Eigen::Quaternionf &q) {
  const Eigen::Matrix3f R = q.toRotationMatrix();
  const Eigen::Vector3f zyx = R.eulerAngles(2, 1, 0); // (z, y, x)
  return pc::types::Float3{rad_to_deg(zyx.z()), rad_to_deg(zyx.y()),
                           rad_to_deg(zyx.x())};
}

bool FloorPlaneAlignmentOperator::compute_floor_euler_degrees(
    const thrust::host_vector<pc::types::position> &positions,
    pc::types::Float3 &out_euler_degrees) {

  pc::logger()->info("Computing floor plane with {} points", positions.size());

  static constexpr float voxel_leaf_mm = 50.0f;
  static constexpr float ransac_distance_threshold_mm = 20.0f;
  static constexpr int ransac_max_iterations = 2000;

  if (positions.size() < 1000) {
    pc::logger()->warn("Point count must be larger than 1000");
    return false;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->points.resize(positions.size());
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const auto &p = positions[i];
    cloud->points[i].x = static_cast<float>(p.x);
    cloud->points[i].y = static_cast<float>(p.y);
    cloud->points[i].z = static_cast<float>(p.z);
  }
  cloud->width = static_cast<std::uint32_t>(cloud->points.size());
  cloud->height = 1;
  cloud->is_dense = true;

  pc::logger()->trace("raw points={}", cloud->size());

  pcl::PointCloud<pcl::PointXYZ>::Ptr voxelised(
      new pcl::PointCloud<pcl::PointXYZ>);
  {
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(voxel_leaf_mm, voxel_leaf_mm, voxel_leaf_mm);
    voxel_grid.filter(*voxelised);
  }

  if (!voxelised || voxelised->empty()) {
    pc::logger()->warn("Failed to compute a valid voxel grid");
    return false;
  }

  pc::logger()->trace("voxelised points={} (leaf={}mm)", voxelised->size(),
                     voxel_leaf_mm);

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(ransac_max_iterations);
  seg.setDistanceThreshold(ransac_distance_threshold_mm);
  seg.setInputCloud(voxelised);

  pcl::PointIndices inliers;
  pcl::ModelCoefficients coeffs;
  seg.segment(inliers, coeffs);

  const std::size_t min_inliers = std::max<std::size_t>(
      500, static_cast<std::size_t>(0.20 * voxelised->size()));

  pc::logger()->trace("inliers={} (min={}) coeffs_n={}", inliers.indices.size(),
                     min_inliers, coeffs.values.size());

  if (inliers.indices.size() < min_inliers) {
    pc::logger()->warn("Not enough inliers to fit plane");
    return false;
  }
  if (coeffs.values.size() < 4) return false;

  Eigen::Vector3f n(coeffs.values[0], coeffs.values[1], coeffs.values[2]);
  const float n_norm = n.norm();
  if (!std::isfinite(n_norm) ||
      n_norm <= std::numeric_limits<float>::epsilon()) {
    pc::logger()->warn("invalid coefficients");
    return false;
  }
  n /= n_norm;

  const Eigen::Vector3f world_up(0.f, 1.f, 0.f);
  if (n.dot(world_up) < 0.f) n = -n;

  const Eigen::Quaternionf q =
      Eigen::Quaternionf::FromTwoVectors(n, world_up).normalized();

  out_euler_degrees = quaternion_to_euler_deg_RzRyRx(q);

  auto clamp360 = [](float d) {
    while (d > 360.f) d -= 360.f;
    while (d < -360.f) d += 360.f;
    return d;
  };
  out_euler_degrees.x = clamp360(out_euler_degrees.x);
  out_euler_degrees.y = clamp360(out_euler_degrees.y);
  out_euler_degrees.z = clamp360(out_euler_degrees.z);

  return true;
}

} // namespace pc::operators::pcl_cpu
