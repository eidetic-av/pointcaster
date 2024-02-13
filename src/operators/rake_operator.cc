#include "rake_operator.gen.h"
#include "../gui/widgets.h"

namespace pc::operators {

  void RakeOperator::draw_imgui_controls(RakeOperatorConfiguration& config) {

    using gui::slider;

    ImGui::PushID(gui::_parameter_index++);

    if (ImGui::CollapsingHeader("Rake", config.unfolded)) {
      config.unfolded = true;
      ImGui::Checkbox("Enabled", &config.enabled);
      
      auto depth_min = (int)config.depth_min_max.min;
	  slider("global_transofmers.rake.depth_min", depth_min, -10000, 10000, 0);
      config.depth_min_max.min = static_cast<short>(depth_min);

      auto depth_max = (int)config.depth_min_max.max;
	  slider("global_transofmers.rake.depth_max", depth_max, -10000, 10000, 0);
      config.depth_min_max.max = static_cast<short>(depth_max);

      auto height_min = (int)config.height_min_max.min;
	  slider("global_transofmers.rake.height_min", height_min, -10000, 10000, 0);
      config.height_min_max.min = static_cast<short>(height_min);

      auto height_max = (int)config.height_min_max.max;
	  slider("global_transofmers.rake.height_max", height_max, -10000, 10000, 0);
      config.height_min_max.max = static_cast<short>(height_max);

    } else {
      config.unfolded = false;
    }

    ImGui::PopID();

  };

}
