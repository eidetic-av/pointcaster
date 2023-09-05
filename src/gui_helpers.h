#pragma once

#include <iostream>

#include "fonts/IconsFontAwesome6.h"
#include "math.h"
#include "structs.h"
#include <array>
#include <atomic>
#include <imgui.h>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using uint = unsigned int;

namespace pc::gui {

enum ParameterType { Float, Int };

/* Must be called at the start of each imgui frame */
void begin_gui_helpers();

extern unsigned int _parameter_index;

struct GuiParameter {
  void *value;
  ParameterType param_type;
  float range_min;
  float range_max;
};

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
  ImGui::PopID();
}

enum class SliderState { Unbound, Bound, Recording };
inline std::unordered_map<std::string, SliderState> slider_states;

struct SliderBinding {
  float value;
  float min = 0.0f;
  float max = 1.0f;
};

inline std::unordered_map<std::string, SliderBinding> slider_bindings;

inline std::atomic_bool recording_slider;
inline std::atomic<SliderState> recording_result;
inline std::string recording_slider_id;
inline std::mutex recording_slider_id_mutex;

inline void store_recording_slider_id(const std::string &id) {
  std::lock_guard<std::mutex> lock(recording_slider_id_mutex);
  recording_slider_id = id;
}

inline std::string load_recording_slider_id() {
  std::lock_guard<std::mutex> lock(recording_slider_id_mutex);
  return recording_slider_id;
}

inline void set_slider_value(std::string slider_id, int value, int input_min,
                             int input_max) {
  auto &binding = slider_bindings[slider_id];
  binding.value = pc::math::remap((float)input_min, (float)input_max,
                                  binding.min, binding.max, (float)value, true);
}

template <typename T>
void slider(const std::string &slider_id, std::size_t i, T &value, T min, T max,
            T reset_value, bool is_disabled, const std::string &label_dimension,
            const std::string &base_label) {
  if (is_disabled)
    ImGui::BeginDisabled();

  auto &state = slider_states[slider_id];

  if (state == SliderState::Bound) {
    auto &binding = slider_bindings[slider_id];
    value = static_cast<T>(
        pc::math::remap(0.0f, 1.0f, (float)min, (float)max, binding.value));
  }

  // if we were recording this slider and we're not anymore,
  // set its status
  if (state == SliderState::Recording && !recording_slider)
    state = recording_result;

  if (state == SliderState::Bound) {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, {0.7f, 0.4f, 0.7f, 0.25f});
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, {0.7f, 0.4f, 0.7f, 0.35f});
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, {0.7f, 0.4f, 0.7f, 0.9f});
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, {0.7f, 0.4f, 0.7f, 1.0f});
  } else if (state == SliderState::Recording) {
    ImGui::PushStyleColor(ImGuiCol_FrameBg, {0.7f, 0.4f, 0.4f, 0.25f});
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, {0.7f, 0.4f, 0.4f, 0.35f});
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, {0.7f, 0.4f, 0.4f, 0.9f});
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, {0.7f, 0.4f, 0.4f, 1.0f});
  }

  // Label Column
  ImGui::TableSetColumnIndex(0);
  ImGui::Text(label_dimension.c_str());

  // Slider Column
  ImGui::TableSetColumnIndex(1);
  ImGui::SetNextItemWidth(-1);

  if (state != SliderState::Bound) {
    if constexpr (std::is_same_v<T, float>) {
      ImGui::SliderFloat(slider_id.c_str(), &value, min, max);
    } else if constexpr (std::is_same_v<T, int>) {
      ImGui::SliderInt(slider_id.c_str(), &value, min, max);
    }
  } else {
    ImGui::SetNextItemWidth(-1);
    auto &binding = slider_bindings[slider_id];
    ImGui::DragFloatRange2(("##" + slider_id + ".minmax").c_str(), &binding.min,
                           &binding.max, 0.001f, 0.0f, 1.0f);
  }

  ImGui::PopStyleColor(state != SliderState::Unbound ? 4 : 0);

  if (!recording_slider && ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
    // Right click toggles controller learning
    if (state == SliderState::Unbound || state == SliderState::Bound) {
      state = SliderState::Recording;
      recording_slider = true;
      store_recording_slider_id(slider_id);
    } else if (state == SliderState::Recording) {
      state = SliderState::Unbound;
      recording_slider = false;
      store_recording_slider_id("");
    }
  }

  // Reset Button Column
  ImGui::TableSetColumnIndex(2);
  if (ImGui::Button("·", {15, 18})) {
    value = reset_value;
  }

  if (is_disabled)
    ImGui::EndDisabled();
}

template <typename T, std::size_t N>
bool vector_table(
    const std::string group_id, const std::string label, std::array<T, N> &vec,
    T min, T max, std::array<T, N> reset_values,
    std::array<bool, N> disabled = std::array<bool, N>{false},
    std::array<std::string, N> labels = std::array<std::string, N>{""}) {
  static_assert(N >= 2 && N <= 4, "Vector array must have 2, 3 or 4 elements");

  constexpr auto outer_horizontal_padding = 4;
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

  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {0, 2});
  ImGui::BeginTable(label.c_str(), 3, ImGuiTableFlags_SizingFixedFit);
  ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthFixed);
  ImGui::TableSetupColumn("##slider", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("##reset_button", ImGuiTableColumnFlags_WidthFixed);

  auto original_vec = vec;

  auto use_labels = !labels.at(0).empty();

  for (std::size_t i = 0; i < N; ++i) {
    const auto &row_label = use_labels ? labels[i] : dimensions[i];
    ImGui::TableNextRow();
    slider(group_id + "." + label + "." + row_label, i, vec[i], min, max,
           reset_values[i], disabled[i], row_label, label);
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
bool vector_table(const std::string group_id, const std::string label,
                  std::array<T, N> &vec, T min, T max, T reset_value) {
  std::array<T, N> reset_values;
  std::fill(reset_values.begin(), reset_values.end(), reset_value);
  return vector_table(group_id, label, vec, min, max, reset_values);
}

// must be inline due to template above also satisfying this overload
inline bool vector_table(const std::string group_id, const std::string label,
                         pc::types::int3 &vec, int min, int max,
                         int reset_value) {
  std::array<int, 3> array_vec = {vec.x, vec.y, vec.z};
  bool result = vector_table(group_id, label, array_vec, min, max, reset_value);
  vec.x = array_vec[0];
  vec.y = array_vec[1];
  vec.z = array_vec[2];
  return result;
}

inline bool vector_table(const std::string group_id, const std::string label,
                         pc::types::int2 &vec, int min, int max,
                         pc::types::int2 reset_values,
                         std::array<bool, 2> disabled = {false, false},
                         std::array<std::string, 2> labels = {"x", "y"}) {
  std::array<int, 2> array_vec = {vec.x, vec.y};
  bool result =
      vector_table(group_id, label, array_vec, min, max,
                   {reset_values.x, reset_values.y}, disabled, labels);
  vec.x = array_vec[0];
  vec.y = array_vec[1];
  return result;
}

inline bool vector_table(const std::string group_id, const std::string label,
                         pc::types::float2 &vec, float min, float max,
                         float reset_value) {
  std::array<float, 2> array_vec = {vec.x, vec.y};
  bool result = vector_table(group_id, label, array_vec, min, max, reset_value);
  vec.x = array_vec[0];
  vec.y = array_vec[1];
  return result;
}

inline bool vector_table(const std::string group_id, const std::string label,
                         pc::types::float2 &vec, float min, float max,
                         pc::types::float2 reset_values,
                         std::array<bool, 2> disabled = {false, false},
                         std::array<std::string, 2> labels = {"x", "y"}) {
  std::array<float, 2> array_vec = {vec.x, vec.y};
  bool result =
      vector_table(group_id, label, array_vec, min, max,
                   {reset_values.x, reset_values.y}, disabled, labels);
  vec.x = array_vec[0];
  vec.y = array_vec[1];
  return result;
}

inline bool vector_table(const std::string group_id, const std::string label,
                         pc::types::float3 &vec, float min, float max,
                         float reset_value) {
  std::array<float, 3> array_vec = {vec.x, vec.y, vec.z};
  bool result = vector_table(group_id, label, array_vec, min, max, reset_value);
  vec.x = array_vec[0];
  vec.y = array_vec[1];
  vec.z = array_vec[2];
  return result;
}

bool begin_tree_node(std::string_view name, bool &open);
} // namespace pc::gui
