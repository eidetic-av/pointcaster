#include "../gui/widgets.h"
#include "../parameters.h"
#include "range_filter_operator.gen.h"
#include "session_bounding_boxes.h"

namespace pc::operators {
void RangeFilterOperator::init(const RangeFilterOperatorConfiguration &config,
                               Scene3D &scene, DrawableGroup3D &parent_group,
                               Vector3 bounding_box_color) {

  set_or_create_bounding_box(config, scene, parent_group,
                             next_bounding_box_color());

  // need to make position and size param updates trigger a bounding
  // box update

  const auto position_parameter = fmt::format("{}.position", config.id);
  const auto size_parameter = fmt::format("{}.size", config.id);

  parameters::add_parameter_update_callback(
      position_parameter, [&](const auto &, auto &) {
        set_or_create_bounding_box(config, scene, parent_group,
                                   next_bounding_box_color());
      });

  parameters::add_parameter_update_callback(
      size_parameter, [&](const auto &, auto &) {
        set_or_create_bounding_box(config, scene, parent_group,
                                   next_bounding_box_color());
      });
}
} // namespace pc::operators
