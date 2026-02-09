#pragma once

#include "../operator.h"

#include <atomic>
#include <mutex>
#include <thread>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/host_vector.h>

namespace pc::operators::pcl_cpu {

using uid = unsigned long int;
using pc::types::Float3;

struct FloorPlaneAlignmentOperatorConfiguration {
  uid id;
  bool unfolded = false;
  bool enabled = true;
  Float3 euler_angles;
};

class FloorPlaneAlignmentOperator {
public:
  explicit FloorPlaneAlignmentOperator(
      FloorPlaneAlignmentOperatorConfiguration &config);

  ~FloorPlaneAlignmentOperator();

  FloorPlaneAlignmentOperator(const FloorPlaneAlignmentOperator &) = delete;
  FloorPlaneAlignmentOperator &
  operator=(const FloorPlaneAlignmentOperator &) = delete;
  FloorPlaneAlignmentOperator(FloorPlaneAlignmentOperator &&) = delete;
  FloorPlaneAlignmentOperator &
  operator=(FloorPlaneAlignmentOperator &&) = delete;

  void update(operator_in_out_t begin, operator_in_out_t end);

  static void
  draw_imgui_controls(FloorPlaneAlignmentOperatorConfiguration &config);

private:
  FloorPlaneAlignmentOperatorConfiguration &_config;

  std::mutex _mutex;
  std::atomic_bool _alignment_thread_finished;
  std::jthread _alignment_thread;

  static bool compute_floor_euler_degrees(
      const thrust::host_vector<pc::types::position> &positions,
      pc::types::Float3 &out_euler_degrees);

  static bool is_in_flight(const pc::types::Float3 &euler_angles);
  static void mark_in_flight(pc::types::Float3 &euler_angles);
};

} // namespace pc::operators::pcl_cpu
