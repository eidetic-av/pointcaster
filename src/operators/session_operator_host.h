#pragma once
#include "../aabb.h"
#include "../gui/catpuccin.h"
#include "../structs.h"
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

namespace pc::operators {

using Scene3D =
    Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>;
using Object3D =
    Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>;

class SessionOperatorHost {

public:
  inline static SessionOperatorHost *instance = nullptr;

  static operator_in_out_t
  run_operators(operator_in_out_t begin, operator_in_out_t end,
                OperatorHostConfiguration &host_config);

  SessionOperatorHost(OperatorHostConfiguration &config, Scene3D &scene,
                      Magnum::SceneGraph::DrawableGroup3D &parent_group);

  void draw_imgui_window();
  void draw_gizmos();

  OperatorHostConfiguration &_config;

  void set_voxel(pc::types::Float3 position,
                 pc::types::Float3 size = {0.05f, 0.05f, 0.05f});
  void set_voxel(pc::types::uid id, pc::types::Float3 position,
                 pc::types::Float3 size = {0.05f, 0.05f, 0.05f});
  void set_cluster(uid id, bool visible, pc::types::Float3 position = {},
                                      pc::types::Float3 size = {0.05f, 0.05f, 0.05f});

private:
  Scene3D &_scene;
  Magnum::SceneGraph::DrawableGroup3D &_parent_group;

  void add_operator(OperatorConfigurationVariant operator_config) {
    _config.operators.push_back(operator_config);
  }

  // std::atomic<std::shared_ptr<std::vector<pc::types::position>>> _pcl_data;
};

using OperatorList =
    std::vector<std::reference_wrapper<const SessionOperatorHost>>;

extern operator_in_out_t apply(operator_in_out_t begin, operator_in_out_t end,
                               const OperatorList &operator_list);

extern pc::types::PointCloud apply(const pc::types::PointCloud &point_cloud,
                                   const OperatorList &operator_list);

} // namespace pc::operators
