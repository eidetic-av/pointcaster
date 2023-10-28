#include "session_operator_host.h"
#include "../logger.h"
#include "../gui/widgets.h"

#include "noise_operator.h"

namespace pc::operators {

void SessionOperatorHost::draw_imgui_window() {
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("Session Operators", nullptr);

  if (ImGui::Button("Add")) {
    _config.operators.push_back(NoiseOperatorConfiguration());
  }

  for (auto &operator_config : _config.operators) {
    std::visit(
	[](auto &&operator_config) {
	  using T = std::decay_t<decltype(operator_config)>;
	  if constexpr (std::is_same_v<T, NoiseOperatorConfiguration>) {
	    NoiseOperator::draw_imgui_controls(operator_config);
	  }
	},
	operator_config);
  }

  ImGui::End();
}

}
