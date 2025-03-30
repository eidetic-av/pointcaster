#pragma once

#include "../objects/solid_box.h"
#include "../objects/wireframe_objects.h"


#include "../gui/catpuccin.h"
#include "../uuid.h"
#include "operator_traits.h"
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

inline std::map<uid, std::unique_ptr<SolidBox>> operator_bounding_boxes;

constexpr std::array<Vector4, 6> bounding_box_colors{
    mocha_red,      mocha_blue,      mocha_yellow,
    mocha_lavender, mocha_rosewater, mocha_peach};

inline Vector3 next_bounding_box_color() {
  static auto current_index = -1;
  current_index = (current_index + 1) % bounding_box_colors.size();
  return bounding_box_colors[current_index].rgb();
}

inline void set_or_create_bounding_box(
    uid id, pc::types::Float3 size, pc::types::Float3 position, Scene3D &scene,
    SceneGraph::DrawableGroup3D &parent_group, bool visible = true,
    std::optional<Color4> color = {}) {
  // get or create the bounding box in the scene
  auto [itr, created_new] = operator_bounding_boxes.emplace(
      id, std::make_unique<SolidBox>(&scene, &parent_group));
  auto &box = itr->second;

  // if it's a new box, set its initial color
  if (created_new) {
    constexpr auto default_transparency = 0.3f;
    box->setColor({color.value_or(next_bounding_box_color()).rgb(),
                   default_transparency});
  }
  box->setVisible(visible);
  // set its updated position / scale
  box->setTransformation(Matrix4::scaling({size.x, size.y, size.z}) *
                         Matrix4::translation({0, 0, 0}));
  box->transform(Matrix4::translation({position.x, position.y, position.z}));
}

template <typename T>
inline void
set_or_create_bounding_box(const T &operator_config, Scene3D &scene,
                           SceneGraph::DrawableGroup3D &parent_group,
                           std::optional<Color4> color = {}) {
  const auto &id = operator_config.id;
  const auto &size = operator_config.transform.size;
  const auto &position = operator_config.transform.position;
  const auto draw = has_draw_v<T> ? operator_config.draw : true;
  set_or_create_bounding_box(id, size, position, scene, parent_group, draw,
                             color);
}

} // namespace pc::operators
