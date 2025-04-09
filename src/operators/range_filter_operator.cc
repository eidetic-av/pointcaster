#include "../gui/widgets.h"
#include "../parameters.h"
#include "range_filter_operator.gen.h"
#include "session_bounding_boxes.h"

namespace pc::operators {

// this function gets a copy of the config with a size that's half the
// specified size. this allows us to serialize the AABB's 'size' but render it
// using its half_size, which is appropriate for an AABB
auto half_extents(auto &&config) {
  auto c = config;
  c.transform.size = {c.transform.size.x / 2, c.transform.size.y / 2,
                      c.transform.size.z / 2};
  return c;
};

void RangeFilterOperator::init(const RangeFilterOperatorConfiguration &config,
                               Scene3D &scene, DrawableGroup3D &parent_group) {
  set_or_create_bounding_box(half_extents(config), scene, parent_group);

  // make position and size param updates trigger
  // bounding box updates

  const auto &id = config.id;
  const auto position_parameter = fmt::format("{}.transform.position", id);
  const auto size_parameter = fmt::format("{}.transform.size", id);

  parameters::add_parameter_update_callback(
      position_parameter, [&](const auto &, auto &) {
        set_or_create_bounding_box(half_extents(config), scene, parent_group);
      });

  parameters::add_parameter_update_callback(
      size_parameter, [&](const auto &, auto &) {
        set_or_create_bounding_box(half_extents(config), scene, parent_group);
      });
}

void RangeFilterOperator::update(const RangeFilterOperatorConfiguration &config,
                                 Scene3D &scene,
                                 DrawableGroup3D &parent_group) {
  set_or_create_bounding_box(half_extents(config), scene, parent_group);
}

} // namespace pc::operators
