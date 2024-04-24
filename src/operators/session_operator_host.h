#pragma once
#include "../structs.h"
#include "operator_host_config.gen.h"
#include <functional>
#include <optional>
#include <vector>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Object.h>

namespace pc::operators {

using Scene3D =
    Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>;
using Object3D =
    Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>;

class SessionOperatorHost {
public:
  static operator_in_out_t run_operators(operator_in_out_t begin,
					 operator_in_out_t end,
					 OperatorHostConfiguration &host_config);

  SessionOperatorHost(OperatorHostConfiguration &config, Scene3D &scene,
		      Magnum::SceneGraph::DrawableGroup3D &parent_group);

  void draw_imgui_window();

  OperatorHostConfiguration &_config;

private:
  Scene3D &_scene;
  Magnum::SceneGraph::DrawableGroup3D &_parent_group;

  void add_operator(OperatorConfigurationVariant operator_config) {
    _config.operators.push_back(operator_config);
  }
};

using OperatorList =
    std::vector<std::reference_wrapper<const SessionOperatorHost>>;

extern operator_in_out_t apply(operator_in_out_t begin, operator_in_out_t end,
			       const OperatorList& operator_list);

extern pc::types::PointCloud apply(const pc::types::PointCloud &point_cloud,
                                   const OperatorList &operator_list);

} // namespace pc::operators
