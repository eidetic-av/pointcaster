#pragma once

#include <imgui.h>
#include <thread>
#include <vector>
#include <string>
#include <string_view>

using uint = unsigned int;

namespace pc::gui {

  enum ParameterType {
    Float, Int
  };

  struct GuiParameter {
    void* value;
    ParameterType param_type;
    float range_min;
    float range_max;
  };

  struct AssignedMidiParameter {
    GuiParameter parameter;
    int channel;
    uint controller_number;
  };

  // extern std::atomic<bool> midi_learn_mode;
  extern bool midi_learn_mode;
  extern std::unique_ptr<GuiParameter> midi_learn_parameter;
  extern std::vector<AssignedMidiParameter> assigned_midi_parameters;

  void enableParameterLearn(void* value_ptr, ParameterType param_type,
			    float range_min = -10, float range_max = 10);

  /* Must be called at the start of each imgui frame */
  void begin_gui_helpers();

  extern unsigned int _parameter_index;

  template <typename T>
  void draw_slider(std::string_view label_text, T *value, T min, T max,
		   T default_value = 0) {
    ImGui::PushID(_parameter_index++);
    ImGui::Text("%s", label_text.data());
    ImGui::SameLine();

    constexpr auto parameter = [](auto text) constexpr {
      return std::string("##") + std::string(text);
    };
    if constexpr (std::is_integral<T>())
      ImGui::SliderInt(parameter(label_text).c_str(), value, min, max);
    else if constexpr (std::is_floating_point<T>())
      ImGui::SliderFloat(parameter(label_text).c_str(), value, min, max);

    ImGui::SameLine();
    if (ImGui::Button("0")) *value = default_value;
    gui::enableParameterLearn(value, gui::ParameterType::Float, min, max);
    ImGui::PopID();
  }
}
