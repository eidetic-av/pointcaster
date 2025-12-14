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
  Float3 max{10000, 10000, 10000}; // @minmax(-10000, 10000)
};

struct RangeFilterOperatorTransformConfiguration {
  Float3 position{0, 1, 0}; // @minmax(-10, 10)
  Float3 size{2, 2, 2};     // @minmax(0.01f, 10)
};

struct RangeFilterOperatorFillConfiguration {
  int fill_count;
  int count_threshold = 5;
  int max_fill = 5000;
  float fill_value;
  float proportion;
};

struct RangeFilterOperatorMinMaxConfiguration {
  bool enabled = false; // @optional
  bool reset_on_empty = true; // @optional
  float min_x;
  float max_x;
  float min_y;
  float max_y;
  float min_z;
  float max_z;
  float volume_change;
  float volume_change_timespan = 0.03f;
};

struct RangeFilterOperatorMinMaxVectorConfiguration {
  Float3 min_x;
  Float3 max_x;
  Float3 min_y;
  Float3 max_y;
  Float3 min_z;
  Float3 max_z;
};

using uid = unsigned long int;

struct RangeFilterOperatorConfiguration {
  uid id;
  bool enabled = true;
  bool bypass = false;
  bool invert = false; // @optional
  bool draw = true;
  RangeFilterOperatorTransformConfiguration transform;
  RangeFilterOperatorFillConfiguration fill;
  RangeFilterOperatorMinMaxConfiguration minmax;
  RangeFilterOperatorMinMaxVectorConfiguration minmax_positions; // @optional
  RangeFilterOperatorMinMaxVectorConfiguration minmax_normalised_positions; // @optional
};

struct RangeFilterOperator : Operator {

  RangeFilterOperatorConfiguration _config;
  Float3 min_mm;
  Float3 max_mm;
  bool invert;

  RangeFilterOperator(const RangeFilterOperatorConfiguration &config)
      : _config(config), invert(config.invert) {
    const auto &position = config.transform.position;
    const auto &size = config.transform.size;
    const Float3 half_size{size.x * 0.5f, size.y * 0.5f, size.z * 0.5f};
    min_mm = Float3{(position.x - half_size.x) * 1000.0f,
                    (position.y - half_size.y) * 1000.0f,
                    (position.z - half_size.z) * 1000.0f};
    max_mm = Float3{(position.x + half_size.x) * 1000.0f,
                    (position.y + half_size.y) * 1000.0f,
                    (position.z + half_size.z) * 1000.0f};
  };

  __device__ __forceinline__ bool operator()(indexed_point_t point) const;

  // TODO these static funcs are a bit random...
  // should be replaced with something outside of the operator
  // since all they really do is control bounding box gizmos
  static void init(const RangeFilterOperatorConfiguration &config,
                   Scene3D &scene, DrawableGroup3D &parent_group);
  static void update(const RangeFilterOperatorConfiguration &config,
                     Scene3D &scene, DrawableGroup3D &parent_group);
};

struct MinMaxXComparator {
  __host__ __device__ bool operator()(indexed_point_t lhs, indexed_point_t rhs) const;
};

struct MinMaxYComparator {
  __host__ __device__  bool operator()(indexed_point_t lhs, indexed_point_t rhs) const;
};

struct MinMaxZComparator {
  __host__ __device__ bool operator()(indexed_point_t lhs, indexed_point_t rhs) const;
};

struct MinMaxXComparer {
  __host__ __device__ bool operator()(const pc::types::position &a,
                                      const pc::types::position &b) {
    return a.x < b.x;
  };
};

struct MinMaxYComparer {
  __host__ __device__ bool operator()(const pc::types::position &a,
                                      const pc::types::position &b) {
    return a.y < b.y;
  };
};

struct MinMaxZComparer {
  __host__ __device__ bool operator()(const pc::types::position &a,
                                      const pc::types::position &b) {
    return a.z < b.z;
  };
};

// this function gets a copy of the config with a size that's half the
// specified size. this allows us to serialize the AABB's 'size' but render it
// using its half_size, which is appropriate for an AABB
inline auto half_extents(auto &&config) {
  auto c = config;
  c.transform.size = {c.transform.size.x / 2, c.transform.size.y / 2,
                      c.transform.size.z / 2};
  return c;
};

} // namespace pc::operators
