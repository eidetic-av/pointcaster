#include "noise_operator.h"
#include "../gui/widgets.h"

namespace pc::operators {

  void NoiseOperator::draw_imgui_controls(NoiseOperatorConfiguration& config) {

    using gui::slider;

    ImGui::PushID(gui::_parameter_index++);

    if (ImGui::CollapsingHeader("Noise", config.unfolded)) {
      config.unfolded = true;
      ImGui::Checkbox("Enabled", &config.enabled);

      slider("global_transformers", "scale", config.scale, 0.0000001f, 0.01f, 0.001f);
      slider("global_transformers", "magnitude", config.magnitude, 0.0f, 1000.0f, 500.0f);
      slider("global_transformers", "seed", config.seed, 1, 1000, 1);
      slider("global_transformers", "repeat", config.repeat, 1, 8, 0);
      slider("global_transformers", "lacunarity", config.lacunarity, 0.001f, 10.0f, 2.0f);
      slider("global_transformers", "decay", config.decay, 0.0f, 1.0f, 0.5f);

    } else {
      config.unfolded = false;
    }

    ImGui::PopID();

  };

}
