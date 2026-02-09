#include "floor_plane_alignment_operator.gen.h"

#include "../../gui/widgets.h"

namespace pc::operators::pcl_cpu {

void FloorPlaneAlignmentOperator::draw_imgui_controls(
    FloorPlaneAlignmentOperatorConfiguration &config) {

  ImGui::PushID(gui::_parameter_index++);

  if (ImGui::CollapsingHeader("Floor Plane Alignment", config.unfolded)) {
    config.unfolded = true;

    ImGui::Checkbox("Enabled", &config.enabled);

    ImGui::BeginDisabled();
    ImGui::InputFloat3("Euler (deg)", &config.euler_angles.x);
    ImGui::EndDisabled();
  } else {
    config.unfolded = false;
  }

  ImGui::PopID();
}

} // namespace pc::operators::pcl_cpu