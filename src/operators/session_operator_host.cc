#include "session_operator_host.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../parameters.h"
#include "../uuid.h"
#include "noise_operator.gen.h"
#include "rake_operator.gen.h"
#include "rotate_operator.gen.h"
#include "sample_filter_operator.gen.h"

namespace pc::operators {

SessionOperatorHost::SessionOperatorHost(OperatorHostConfiguration &config)
    : _config(config) {

  // for loading an existing list of operators
  for (auto &operator_config : _config.operators) {
    // we need to visit each possible operator variant
    std::visit(
        [](auto &&operator_instance) {
	  // and declare it's saved ID with its parameters
	  declare_parameters(std::to_string(operator_instance.id),
			     operator_instance);
        },
        operator_config);
  }
};

void SessionOperatorHost::draw_imgui_window() {
  ImGui::SetNextWindowSize({600, 400}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Session Operators", nullptr);

  static bool select_node = false;

  if (ImGui::Button("Add session operator")) {
    ImGui::OpenPopup("Add session operator");
  }

  if (ImGui::BeginPopup("Add session operator")) {
    // populate menu with all Operator types

    apply_to_all_operators([this](auto &&operator_type) {
      using T = std::remove_reference_t<decltype(operator_type)>;
      if (ImGui::Selectable(T::Name)) {

        auto operator_config = T();
        operator_config.id = pc::uuid::digit();
	// create a new instance of our operator configuration and add it to our
	// session operator list
	auto &variant_ref = _config.operators.emplace_back(operator_config);

	// declare the instance as parameters to bind to this new operator's id
	auto &config_instance = std::get<T>(variant_ref);
	declare_parameters(std::to_string(operator_config.id), config_instance);

        ImGui::CloseCurrentPopup();
      }
    });

    ImGui::EndPopup();
  }

  for (auto &operator_config : _config.operators) {
    ImGui::PushID(gui::_parameter_index++);
    std::visit(
	[](auto &&config) {
	  using T = std::decay_t<decltype(config)>;

	  ImGui::PushID(gui::_parameter_index++);
          if (ImGui::CollapsingHeader(T::Name, config.unfolded)) {
	    config.unfolded = true;
	    pc::gui::draw_parameters(config.id);
          } else {
            config.unfolded = false;
          }
	  ImGui::PopID();
        },
        operator_config);
    ImGui::PopID();
  }

  ImGui::End();
}

} // namespace pc::operators
