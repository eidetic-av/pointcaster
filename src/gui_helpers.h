#pragma once

#include <imgui.h>
#include <thread>
#include <vector>

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
}
