#include "rotate_operator.gen.h"
#include "../gui/widgets.h"

namespace pc::operators {

  void RotateOperator::draw_imgui_controls(RotateOperatorConfiguration& config) {

    using gui::vector_param;

    ImGui::PushID(gui::_parameter_index++);

    if (ImGui::CollapsingHeader("Rotate", config.unfolded)) {
      config.unfolded = true;
      ImGui::Checkbox("Enabled", &config.enabled);

      vector_param(std::to_string(gui::_parameter_index), "global_transformers/rotate", config.euler_angles, 0.0f, 360.f, 0.0f);

    } else {
      config.unfolded = false;
    }

    ImGui::PopID();

  };

}
