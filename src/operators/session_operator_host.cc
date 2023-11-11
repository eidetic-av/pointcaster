#include "session_operator_host.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "knn_filter_operator.h"
#include "noise_operator.h"
#include "operators.h"

namespace pc::operators {

void SessionOperatorHost::draw_imgui_window() {
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::Begin("Session Operators", nullptr);

  if (ImGui::Button("Add session operator")) {
    ImGui::OpenPopup("Add session operator");
  }

  if (ImGui::BeginPopup("Add session operator")) {
    // populate menu with all Operator types

    apply_to_all_operators([this](auto &&operator_type) {
      using T = std::remove_reference_t<decltype(operator_type)>;
      if (ImGui::Selectable(T::Name)) {
	_config.operators.push_back(T());
        ImGui::CloseCurrentPopup();
      }
    });

    ImGui::EndPopup();
  }

  for (auto &operator_config : _config.operators) {
    std::visit(
	[](auto &&operator_config) {

	  using T = std::decay_t<decltype(operator_config)>;

	  if constexpr (std::is_same_v<T, NoiseOperatorConfiguration>) {
	    NoiseOperator::draw_imgui_controls(operator_config);
	  }

	  if constexpr (std::is_same_v<T, KNNFilterOperatorConfiguration>) {
	    KNNFilterOperator::draw_imgui_controls(operator_config);
	  }

	},
	operator_config);
  }

  ImGui::End();
}

}
