#include "sample_filter_operator.gen.h"
#include "../gui/widgets.h"

namespace pc::operators {

  void SampleFilterOperator::draw_imgui_controls(SampleFilterOperatorConfiguration& config) {

    using gui::slider;

    ImGui::PushID(gui::_parameter_index++);

    if (ImGui::CollapsingHeader("Sample Filter", config.unfolded)) {
      config.unfolded = true;
      ImGui::Checkbox("Enabled", &config.enabled);

      ImGui::TextDisabled("Sample count");
      slider("session_operators", "sample_count", config.sample_count, 1, 50, 1);

    } else {
      config.unfolded = false;
    }

    ImGui::PopID();

  };

}
