#pragma once

#include <Magnum/Math/Half.h>

#include "../aabb.h"
#include "../gui/catpuccin.h"
#include "../structs.h"
#include "operator_friendly_names.h"
#include "operator_host_config.gen.h"
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Object.h>
#include <atomic>
#include <functional>
#include <optional>
#include <pcl/point_cloud.h>
#include <tbb/concurrent_queue.h>
#include <tbb/parallel_pipeline.h>
#include <thread>
#include <thrust/host_vector.h>
#include <vector>
#include <string_view>

// opaque but full header included inside session_operator_host.cc
namespace pc {
class PointCaster;
}

namespace pc::operators {

using Scene3D =
    Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>;
using Object3D =
    Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>;

class SessionOperatorHost {

public:
  inline static SessionOperatorHost *instance = nullptr;

  const std::string session_id;
  const uid session_seed;
  OperatorHostConfiguration _config;

  static operator_in_out_t run_operators(operator_in_out_t begin,
                                         operator_in_out_t end,
                                         OperatorHostConfiguration &host_config,
                                         const std::string_view session_id);

  static operator_in_out_t run_operators(
      operator_in_out_t begin, operator_in_out_t end,
      std::vector<OperatorConfigurationVariant> &operator_variant_list,
      const std::string_view session_id);

  explicit SessionOperatorHost(
      pc::PointCaster &app, Scene3D &scene,
      Magnum::SceneGraph::DrawableGroup3D &parent_group,
      std::string_view host_session_id, OperatorHostConfiguration config = {});

  void draw_imgui_window();
  void draw_gizmos();
  void clear_gizmos();

  void set_voxel(pc::types::Float3 position,
                 pc::types::Float3 size = {0.05f, 0.05f, 0.05f});
  void set_voxel(pc::types::uid id, bool visible,
                 pc::types::Float3 position = {},
                 pc::types::Float3 size = {0.05f, 0.05f, 0.05f});
  void set_cluster(uid id, bool visible, pc::types::Float3 position = {},
                   pc::types::Float3 size = {0.05f, 0.05f, 0.05f});

private:
  pc::PointCaster &_app;
  Scene3D &_scene;
  Magnum::SceneGraph::DrawableGroup3D &_parent_group;
  std::optional<size_t> _last_voxel_count;
  std::optional<size_t> _last_cluster_count;

  void add_operator(OperatorConfigurationVariant operator_config) {
    _config.operators.push_back(operator_config);
  }

  // std::atomic<std::shared_ptr<std::vector<pc::types::position>>> _pcl_data;
};

} // namespace pc::operators
