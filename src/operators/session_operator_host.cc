#include "session_operator_host.h"
#include "../logger.h"
#include "../gui/widgets.h"

#include "noise_operator.h"
#include "rotate_operator.h"
#include "rake_operator.h"

namespace pc::operators {

void SessionOperatorHost::draw_imgui_window() {
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("Session Operators", nullptr);

  if (ImGui::Button("Noise")) {
    _config.operators.push_back(NoiseOperatorConfiguration());
  }
  ImGui::SameLine();
  if (ImGui::Button("Rotate")) {
    _config.operators.push_back(RotateOperatorConfiguration());
  }
  ImGui::SameLine();
  if (ImGui::Button("Rake")) {
    _config.operators.push_back(RakeOperatorConfiguration());
  }

  for (auto &operator_config : _config.operators) {
    std::visit(
	[](auto &&operator_config) {
	  using T = std::decay_t<decltype(operator_config)>;
	  if constexpr (std::is_same_v<T, NoiseOperatorConfiguration>) {
	    NoiseOperator::draw_imgui_controls(operator_config);
	  }
	  else if constexpr (std::is_same_v<T, RotateOperatorConfiguration>) {
	    RotateOperator::draw_imgui_controls(operator_config);
	  }
	  else if constexpr (std::is_same_v<T, RakeOperatorConfiguration>) {
	    RakeOperator::draw_imgui_controls(operator_config);
	  }
	},
	operator_config);
  }

  ImGui::End();
}

}
