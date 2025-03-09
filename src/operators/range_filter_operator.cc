#include "../gui/widgets.h"
#include "../parameters.h"
#include "range_filter_operator.gen.h"
#include "session_bounding_boxes.h"

namespace pc::operators {
void RangeFilterOperator::init(const RangeFilterOperatorConfiguration &config,
                               Scene3D &scene, DrawableGroup3D &parent_group,
                               Vector3 bounding_box_color) {

  // TODO
  const auto color = next_bounding_box_color();

  set_or_create_bounding_box(config, scene, parent_group, color);

  // need to make position and size param updates trigger a bounding
  // box update

  const auto &id = config.id;
  const auto position_parameter = fmt::format("{}.transform.position", id);
  const auto size_parameter = fmt::format("{}.transform.size", id);

  parameters::add_parameter_update_callback(
      position_parameter, [&, color](const auto &, auto &) {
        set_or_create_bounding_box(config, scene, parent_group, color);
      });

  parameters::add_parameter_update_callback(
      size_parameter, [&, color](const auto &, auto &) {
        set_or_create_bounding_box(config, scene, parent_group, color);
      });
}
} // namespace pc::operators
