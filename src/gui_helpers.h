#pragma once

#include <imgui.h>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <array>
#include "structs.h"
#include "fonts/IconsFontAwesome6.h"

using uint = unsigned int;

namespace pc::gui {

enum ParameterType { Float, Int };

struct GuiParameter {
  void *value;
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

void enableParameterLearn(void *value_ptr, ParameterType param_type,
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
    ImGui::SliderFloat(parameter(label_text).c_str(), value, min, max, "%.5g");

  ImGui::SameLine();
  if (ImGui::Button("0"))
    *value = default_value;
  gui::enableParameterLearn(value, gui::ParameterType::Float, min, max);
  ImGui::PopID();
}

template <typename T, std::size_t N>
bool vector_table(const std::string &label, std::array<T, N> &vec, T min, T max,
                  std::array<T, N> reset_values,
                  std::array<bool, N> disabled = std::array<bool, N>{false},
                  std::array<std::string, N> labels = std::array<std::string, N>{""}) {
  static_assert(N >= 2 && N <= 4, "Vector array must have 2, 3 or 4 elements");

  constexpr auto outer_horizontal_padding = 4;
  constexpr auto label_padding_left = 3;
  constexpr auto label_padding_right = 6;
  constexpr auto label_padding_top = 0;
  constexpr auto table_background_color = IM_COL32(22, 27, 34, 255);

  const char *dimensions[] = {"x", "y", "z", "w"};

  auto row_height = ImGui::GetTextLineHeightWithSpacing() * 1.33f;
  auto table_height = (row_height * (N + 1)) + 14;

  ImGui::PushID(_parameter_index++);

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                      {outer_horizontal_padding, 0});
  ImGui::PushStyleColor(ImGuiCol_ChildBg, table_background_color);
  ImGui::BeginChild("##vector_table_border", {0, table_height}, true,
                    ImGuiWindowFlags_AlwaysAutoResize |
                        ImGuiWindowFlags_NoScrollbar);
  ImGui::Dummy({0, outer_horizontal_padding});
  ImGui::Dummy({outer_horizontal_padding, 0});
  ImGui::SameLine(0, 0);
  ImGui::Text(label.c_str());
  ImGui::Dummy({0, 0});

  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {0, 0});
  ImGui::BeginTable(label.c_str(), 3, ImGuiTableFlags_SizingFixedFit);
  ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed);
  ImGui::TableSetupColumn("##slider", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("##reset_button", ImGuiTableColumnFlags_WidthFixed);

  auto original_vec = vec;

  auto use_labels = !labels.at(0).empty();

  for (std::size_t i = 0; i < N; ++i) {
    if (disabled[i]) ImGui::BeginDisabled();
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Dummy({0, label_padding_top});
    ImGui::Dummy({label_padding_left, 0});
    ImGui::SameLine(0, 0);
    if (use_labels) ImGui::Text(labels[i].c_str());
    else ImGui::Text(dimensions[i]);
    ImGui::SameLine(0, 0);
    ImGui::Dummy({label_padding_right, 0});
    ImGui::TableSetColumnIndex(1);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
    ImGui::SetNextItemWidth(-1);
    if constexpr (std::is_same_v<T, float>) {
      ImGui::SliderFloat(
          (std::string("##") + label + "." + dimensions[i]).c_str(), &vec[i],
          min, max);
    } else if constexpr (std::is_same_v<T, int>) {
      ImGui::SliderInt(
          (std::string("##") + label + "." + dimensions[i]).c_str(), &vec[i],
          min, max);
    }
    ImGui::Dummy({0, 5});
    ImGui::PopStyleVar();
    ImGui::TableSetColumnIndex(2);
    ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
    ImGui::Dummy({0, 1});
    ImGui::PushID(_parameter_index++);
    if (ImGui::Button("·", {15, 18})) {
      vec[i] = reset_values[i];
    }
    ImGui::PopID();
    ImGui::SameLine(0, 0);
    ImGui::Dummy({0, 0});
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    if (disabled[i]) ImGui::EndDisabled();
  }

  ImGui::EndTable();
  ImGui::PopStyleVar();

  ImGui::EndChild();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();

  ImGui::PopID();

  return vec != original_vec;
}

template <typename T, std::size_t N>
bool vector_table(const std::string &label, std::array<T, N> &vec, T min, T max,
		  T reset_value) {
  std::array<T, N> reset_values;
  std::fill(reset_values.begin(), reset_values.end(), reset_value);
  return vector_table(label, vec, min, max, reset_values);
}

// must be inline due to template above also satisfying this overload
inline bool vector_table(const std::string &label, pc::types::int3 &vec,
			 int min, int max, int reset_value) {
    std::array<int, 3> array_vec = {vec.x, vec.y, vec.z};
    bool result = vector_table(label, array_vec, min, max, reset_value);
    vec.x = array_vec[0];
    vec.y = array_vec[1];
    vec.z = array_vec[2];
    return result;
}

inline bool vector_table(const std::string &label, pc::types::float3 &vec,
			 float min, float max, float reset_value) {
    std::array<float, 3> array_vec = {vec.x, vec.y, vec.z};
    bool result = vector_table(label, array_vec, min, max, reset_value);
    vec.x = array_vec[0];
    vec.y = array_vec[1];
    vec.z = array_vec[2];
    return result;
}

bool begin_tree_node(std::string_view name, bool &open);
} // namespace pc::gui
