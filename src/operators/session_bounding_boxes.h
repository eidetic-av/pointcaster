#pragma once

#include "../gui/catpuccin.h"
#include "../uuid.h"
#include "../wireframe_objects.h"
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

inline static std::map<uid, std::unique_ptr<WireframeBox>>
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
      operator_config.id,
      std::make_unique<WireframeBox>(&scene, &parent_group));
  auto &box = itr->second;
  // and set its updated position / scale
  const auto &size = operator_config.size;
  const auto &position = operator_config.position;
  box->set_transformation(Matrix4::scaling({size.x, size.y, size.z}) *
                          Matrix4::translation({0, 0, 0}));
  box->transform(Matrix4::translation({position.x, position.y, position.z}));
  if (color.has_value()) {
    box->set_color(color.value().rgb());
  }
  if constexpr (std::same_as<T, RangeFilterOperatorConfiguration>) {
    box->set_visible(operator_config.draw);
  }
}

static void
set_or_create_bounding_box(uid id, pc::types::Float3 size, pc::types::Float3 position,
	Scene3D& scene, SceneGraph::DrawableGroup3D& parent_group, std::optional<Color4> color = {}) {
	// get or create the bounding box in the scene
	auto [itr, _] = operator_bounding_boxes.emplace(
		id, std::make_unique<WireframeBox>(&scene, &parent_group));
	auto& box = itr->second;
	// and set its updated position / scale
	box->set_transformation(Matrix4::scaling({ size.x, size.y, size.z }) *
		Matrix4::translation({ 0, 0, 0 }));

	box->transform(Matrix4::translation({ position.x, position.y, position.z }));

	if (color.has_value()) {
		box->set_color(color.value().rgb());
	}
	else {
		box->set_color(next_bounding_box_color());
	}
	box->set_visible(true);
}

} // namespace pc::operators
