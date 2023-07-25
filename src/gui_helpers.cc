#include "gui_helpers.h"
#include <spdlog/spdlog.h>

namespace pc::gui {

  // std::atomic<bool> midi_learn_mode = false;
  bool midi_learn_mode = false;
  std::unique_ptr<GuiParameter> midi_learn_parameter;
  std::vector<AssignedMidiParameter> assigned_midi_parameters;

  void enableParameterLearn(void *value_ptr, ParameterType param_type,
			    float range_min, float range_max) {
    if (!midi_learn_mode) return;
    if (ImGui::IsItemClicked()) {
      midi_learn_parameter.reset(
	  new GuiParameter{value_ptr, param_type, range_min, range_max});
    }
  }
}
