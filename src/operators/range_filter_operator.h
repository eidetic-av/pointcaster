#pragma once
#include "../serialization.h"
#include "../structs.h"
#include "operator.h"
#include <Magnum/Magnum.h>
#include <Magnum/SceneGraph/SceneGraph.h>

namespace pc::operators {

using pc::types::Float2;
using pc::types::Float3;

using Object3D =
    Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>;
using Scene3D =
    Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>;
using Magnum::Vector3;
using Magnum::SceneGraph::DrawableGroup3D;

struct AABB {
  bool unfolded = true;
  Float3 min{-10000, -10000, -10000}; // @minmax(-10000, 10000)
  Float3 max{10000, 10000, 10000};    // @minmax(-10000, 10000)
};

using uid = unsigned long int;

struct RangeFilterOperatorFillConfiguration {
  int fill_count;
  int count_threshold = 5;
  int max_fill = 5000;
  float fill_value;
  float proportion;
  bool publish = false;
};

struct RangeFilterOperatorMinMaxConfiguration {
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float min_z;
  float max_z;
  float volume_change;
  float volume_change_timespan = 0.03f;
  bool publish = false;
};

struct RangeFilterOperatorConfiguration {
  uid id;
  bool enabled = true;
  bool bypass = false;
  bool draw = true;
  Float3 position{0, 0, 0}; // @minmax(-10, 10)
  Float3 size{1, 1, 1};     // @minmax(0.01f, 10)
  RangeFilterOperatorFillConfiguration fill;
  RangeFilterOperatorMinMaxConfiguration minmax;
};

struct RangeFilterOperator : Operator {

  RangeFilterOperatorConfiguration _config;

  RangeFilterOperator(const RangeFilterOperatorConfiguration &config)
      : _config(config) {};

  __device__ bool operator()(indexed_point_t point) const;

  static void init(const RangeFilterOperatorConfiguration &config,
                   Scene3D &scene, DrawableGroup3D &parent_group,
                   Vector3 bounding_box_color);
};

struct MinMaxXComparator {
  __device__ bool operator()(indexed_point_t lhs, indexed_point_t rhs) const;
};

struct MinMaxYComparator {
  __device__ bool operator()(indexed_point_t lhs, indexed_point_t rhs) const;
};

struct MinMaxZComparator {
  __device__ bool operator()(indexed_point_t lhs, indexed_point_t rhs) const;
};

} // namespace pc::operators
