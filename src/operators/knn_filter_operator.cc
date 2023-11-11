#include "knn_filter_operator.h"
#include "../gui/widgets.h"

namespace pc::operators {

  void KNNFilterOperator::draw_imgui_controls(KNNFilterOperatorConfiguration& config) {

    using gui::slider;

    ImGui::PushID(gui::_parameter_index++);

    if (ImGui::CollapsingHeader("KNN Filter", config.unfolded)) {
      config.unfolded = true;
      ImGui::Checkbox("Enabled", &config.enabled);

    } else {
      config.unfolded = false;
    }

    ImGui::PopID();

  };

}
