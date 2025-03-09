#pragma once

#include "../objects/wireframe_objects.h"
#include "../objects/solid_objects.h"

#include "../gui/catpuccin.h"
#include "../uuid.h"
#include "range_filter_operator.gen.h"
#include <Magnum/Magnum.h>
#include <map>
#include <memory>
#include <optional>
#include <random>

namespace pc::operators {

using namespace catpuccin;
using namespace catpuccin::magnum;
using Vector4 = Magnum::Math::Vector4<float>;

using uid = unsigned long int;

inline static std::map<uid, std::unique_ptr<SolidBox>>
    operator_bounding_boxes;

inline static constexpr std::array<Vector4, 6> bounding_box_colors{
    mocha_red,      mocha_blue,      mocha_yellow,
    mocha_lavender, mocha_rosewater, mocha_peach};

inline static Vector3 next_bounding_box_color() {
  static auto current_index = -1;
  current_index = (current_index + 1) % bounding_box_colors.size();
  return bounding_box_colors[current_index].rgb();
}

template <typename T>
static void
set_or_create_bounding_box(const T &operator_config, Scene3D &scene,
                           SceneGraph::DrawableGroup3D &parent_group,
                           std::optional<Color4> color = {}) {
  // get or create the bounding box in the scene
  auto [itr, _] = operator_bounding_boxes.emplace(
      operator_config.id, std::make_unique<SolidBox>(&scene, &parent_group));

  auto &box = itr->second;
  // and set its updated position / scale
  const auto &size = operator_config.transform.size;
  const auto &position = operator_config.transform.position;
  box->setTransformation(Matrix4::scaling({size.x, size.y, size.z}) *
                         Matrix4::translation({0, 0, 0}));
  box->transform(Matrix4::translation({position.x, position.y, position.z}));

  if (color.has_value()) {
    auto rgb = color.value().rgb();
    box->setColor({rgb, 0.3f});
  } else {
    auto rgb = next_bounding_box_color();
    box->setColor({rgb, 0.3f});
  }

  if constexpr (std::same_as<T, RangeFilterOperatorConfiguration>) {
    // box->set_visible(operator_config.draw);
  }
}

static void
set_or_create_bounding_box(uid id, pc::types::Float3 size,
                           pc::types::Float3 position, Scene3D &scene,
                           SceneGraph::DrawableGroup3D &parent_group,
                           std::optional<Color4> color = {}) {
  // get or create the bounding box in the scene
  auto [itr, _] = operator_bounding_boxes.emplace(
      id, std::make_unique<SolidBox>(&scene, &parent_group));
  auto &box = itr->second;
  // and set its updated position / scale
  box->setTransformation(Matrix4::scaling({size.x, size.y, size.z}) *
                         Matrix4::translation({0, 0, 0}));
  box->transform(Matrix4::translation({position.x, position.y, position.z}));

  if (color.has_value()) {
    auto rgb = color.value().rgb();
    box->setColor({rgb, 0.3f});
  } else {
    auto rgb = next_bounding_box_color();
    box->setColor({rgb, 0.3f});
  }
}

} // namespace pc::operators
