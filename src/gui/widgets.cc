#include "widgets.h"
#include "../logger.h"
#include "../string_utils.h"
#include <array>
#include <cmath>
#include <tweeny/easing.h>

namespace pc::gui {

unsigned int _parameter_index;

void begin_gui_helpers() { _parameter_index = 0; }

constexpr std::string format_label(std::string_view label) {
  return strings::sentence_case(strings::last_element(label));
};

static bool inside_widget_container = false;

void begin_widget_container(std::string_view widget_label = "",
			    std::size_t row_count = 0) {
  constexpr auto outer_horizontal_padding = 4;
  constexpr auto table_background_color = IM_COL32(22, 27, 34, 255);

  const auto parameter_index_string = std::to_string(_parameter_index);

  const auto row_height = ImGui::GetTextLineHeightWithSpacing() * 1.33f;
  const auto table_height = (row_height * (row_count + 1)) + 14;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
		      {outer_horizontal_padding, 0});
  ImGui::PushStyleColor(ImGuiCol_ChildBg, table_background_color);

  const auto child_id = "##widget_container_border." + parameter_index_string;
  ImGui::BeginChild(child_id.data(), {0, table_height}, true,
		    ImGuiWindowFlags_AlwaysAutoResize |
			ImGuiWindowFlags_NoScrollbar);

  ImGui::Dummy({0, outer_horizontal_padding});
  ImGui::Dummy({outer_horizontal_padding, 0});
  ImGui::SameLine(0, 0);

  if (!widget_label.empty()) {
    ImGui::Text("%s", format_label(widget_label).data());
    ImGui::Dummy({0, 0});
  }

  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {0, 2});

  const auto table_id = "##table." + std::to_string(_parameter_index);
  ImGui::BeginTable(table_id.c_str(), 3, ImGuiTableFlags_SizingFixedFit);

  ImGui::TableSetupColumn("##label", ImGuiTableColumnFlags_WidthStretch, 0.3f);
  ImGui::TableSetupColumn("##slider", ImGuiTableColumnFlags_WidthStretch, 0.7f);
  ImGui::TableSetupColumn("##reset_button", ImGuiTableColumnFlags_WidthFixed);

  inside_widget_container = true;
}

void end_widget_container() {
  ImGui::EndTable();
  ImGui::PopStyleVar();
  ImGui::EndChild();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();

  inside_widget_container = false;
}

template <typename T>
bool slider(std::string_view parameter_id, T &value, T min, T max,
	    T reset_value, bool is_disabled, std::string_view label) {

  bool standalone_widget = !inside_widget_container;
  if (standalone_widget) {
    begin_widget_container();
  }

  ImGui::TableNextRow();

  ImGui::PushID(_parameter_index++);

  if (is_disabled) ImGui::BeginDisabled();

  auto &state = parameter_states[parameter_id];
  auto new_state = state;

  if (state == ParameterState::Bound) {
    // colour purple if the slider is a bound parameter
    ImGui::PushStyleColor(ImGuiCol_FrameBg, {0.7f, 0.4f, 0.7f, 0.25f});
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, {0.7f, 0.4f, 0.7f, 0.35f});
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, {0.7f, 0.4f, 0.7f, 0.9f});
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, {0.7f, 0.4f, 0.7f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_Button, {0.7f, 0.4f, 0.7f, 0.25f});
  } else if (state == ParameterState::Learning) {
    // colour it red for recording
    ImGui::PushStyleColor(ImGuiCol_FrameBg, {0.7f, 0.4f, 0.4f, 0.25f});
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, {0.7f, 0.4f, 0.4f, 0.35f});
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, {0.7f, 0.4f, 0.4f, 0.9f});
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, {0.7f, 0.4f, 0.4f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_Button, {0.7f, 0.4f, 0.4f, 0.25f});
    // if we *were* recording this slider and we're not anymore,
    // set its status
    if (!learning_parameter) new_state = recording_result;
  }

  // Label Column
  ImGui::TableSetColumnIndex(0);
  std::string label_text;
  if (label.empty()) {
    label_text = format_label(parameter_id).data();
  } else {
    label_text = label;
  }

  // align text right
  auto pos_x = (ImGui::GetCursorPosX() + ImGui::GetColumnWidth() -
		ImGui::CalcTextSize(label_text.c_str()).x -
		ImGui::GetScrollX() - 2 * ImGui::GetStyle().ItemSpacing.x);
  if (pos_x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(pos_x);

  ImGui::Text("%s", label_text.data());

  // Slider Column
  ImGui::TableSetColumnIndex(1);
  ImGui::SetNextItemWidth(-1);

  bool updated = false;

  if (state != ParameterState::Bound) {
    if constexpr (std::is_same_v<T, float>) {
      updated = ImGui::SliderFloat(parameter_id.data(), &value, min, max, "%.5g");
    } else if constexpr (std::is_same_v<T, int>) {
      updated = ImGui::SliderInt(parameter_id.data(), &value, min, max);
    }
  } else {
    // if the slider is bound, draw a range slider to set the min and max values
    ImGui::SetNextItemWidth(-1);
    auto &binding = parameter_bindings.at(parameter_id);
    auto old_binding = binding;
    if (ImGui::RangeSliderFloat(("##" + std::string(parameter_id) + ".minmax").data(),
                                &binding.min, &binding.max, min, max)) {
      // if the mapping range is updated, update the value itself to the new
      // range
      if (std::holds_alternative<FloatReference>(binding.value)) {
        float &value = std::get<FloatReference>(binding.value).get();
        float &old_value = std::get<FloatReference>(old_binding.value).get();
        value = math::remap(old_binding.min, old_binding.max, binding.min,
                            binding.max, old_value);
      } else {
        int &value = std::get<IntReference>(binding.value).get();
        int &old_value = std::get<IntReference>(old_binding.value).get();
        auto float_value =
            math::remap(old_binding.min, old_binding.max, binding.min,
                        binding.max, static_cast<float>(old_value));
        value = static_cast<int>(std::round(float_value));
      }
      // and trigger any update callbacks
      for (const auto &cb : binding.minmax_update_callbacks) {
        cb(old_binding, binding);
      }
    }
  }

  bool unbind_current = false;

  // Right click toggles controller learning
  if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
    if (!learning_parameter) {
      // if we were not recording, set it to record
      new_state = ParameterState::Learning;
      learning_parameter = true;
      store_learning_parameter_info(parameter_id, min, max, value);
    } else {
      // if we were recording, return the slider to an unbound state
      new_state = ParameterState::Unbound;
      learning_parameter = false;
      unbind_current = true;
    }
  }

  // Reset Button Column
  ImGui::TableSetColumnIndex(2);
  if (state == ParameterState::Bound) ImGui::BeginDisabled();
  if (ImGui::Button("·", {15, 18})) {
    value = reset_value;
    updated = true;
  }
  if (state == ParameterState::Bound) ImGui::EndDisabled();

  ImGui::PopStyleColor(state != ParameterState::Unbound ? 5 : 0);

  if (is_disabled) ImGui::EndDisabled();

  state = new_state;

  if (unbind_current) unbind_parameter(parameter_id);

  ImGui::PopID();

  if (standalone_widget) end_widget_container();

  return updated;
}

template bool slider(std::string_view parameter_id, float &value, float min,
		     float max, float reset_value, bool is_disabled,
		     std::string_view label);

template bool slider(std::string_view parameter_id, int &value, int min,
                     int max, int reset_value, bool is_disabled,
                     std::string_view label);

template <typename T>
bool vector_table(
    std::string_view group_id, std::string_view parameter_id, T &vec,
    typename T::vector_type min, typename T::vector_type max,
    std::array<typename T::vector_type, types::VectorSize<T>::value>
        reset_values,
    std::array<bool, types::VectorSize<T>::value> disabled,
    std::array<std::string, types::VectorSize<T>::value> labels) {

  constexpr auto vector_size = types::VectorSize<T>::value;

  ImGui::PushID(_parameter_index++);

  begin_widget_container(parameter_id, vector_size);

  auto original_vec = vec;

  constexpr std::array<const char *, 4> elements = {"x", "y", "z", "w"};

  auto use_labels = !labels.at(0).empty();

  for (std::size_t i = 0; i < vector_size; ++i) {
    const auto &row_label = use_labels ? labels[i] : elements[i];
    auto vector_parameter_id =
	fmt::format("{}.{}.{}", group_id, parameter_id, elements[i]);
    slider(vector_parameter_id, vec[i], min, max, reset_values[i], disabled[i],
	   row_label);
  }

  end_widget_container();

  ImGui::PopID();

  return vec != original_vec;
}

template bool vector_table(std::string_view group_id,
                           std::string_view parameter_id, int2 &vec, int min,
                           int max, std::array<int, 2> reset_values,
                           std::array<bool, 2> disabled,
                           std::array<std::string, 2> labels);

template bool vector_table(std::string_view group_id,
                           std::string_view parameter_id, int3 &vec, int min,
                           int max, std::array<int, 3> reset_values,
                           std::array<bool, 3> disabled,
                           std::array<std::string, 3> labels);

template bool vector_table(std::string_view group_id,
                           std::string_view parameter_id, float2 &vec,
                           float min, float max,
                           std::array<float, 2> reset_values,
                           std::array<bool, 2> disabled,
                           std::array<std::string, 2> labels);

template bool vector_table(std::string_view group_id,
                           std::string_view parameter_id, float3 &vec,
                           float min, float max,
                           std::array<float, 3> reset_values,
                           std::array<bool, 3> disabled,
                           std::array<std::string, 3> labels);

template bool vector_table(std::string_view group_id,
                           std::string_view parameter_id, float4 &vec,
                           float min, float max,
                           std::array<float, 4> reset_values,
                           std::array<bool, 4> disabled,
                           std::array<std::string, 4> labels);

bool begin_tree_node(std::string_view name, bool &open) {
  ImGui::PushID(_parameter_index++);
  auto node_flags = ImGuiTreeNodeFlags_None;
  if (open)
    node_flags = ImGuiTreeNodeFlags_DefaultOpen;
  open = ImGui::TreeNodeEx(name.data(), node_flags);
  ImGui::PopID();
  return open;
}

void tween_config(std::string_view group_id, std::string_view parameter_id,
		  pc::tween::TweenConfiguration &config) {
  ImGui::PushID(_parameter_index++);

  slider(parameter_id, config.duration_ms, 0, 2000, 300);

  static const std::map<tweeny::easing::enumerated, std::pair<int, std::string>>
      ease_function_to_combo_item = {
          {tweeny::easing::enumerated::def, {0, "Default"}},
          {tweeny::easing::enumerated::linear, {1, "Linear"}},
          {tweeny::easing::enumerated::stepped, {2, "Stepped"}},
          {tweeny::easing::enumerated::quadraticIn, {3, "Quadratic In"}},
          {tweeny::easing::enumerated::quadraticOut, {4, "Quadratic Out"}},
          {tweeny::easing::enumerated::quadraticInOut, {5, "Quadratic InOut"}},
          {tweeny::easing::enumerated::cubicIn, {6, "Cubic In"}},
          {tweeny::easing::enumerated::cubicOut, {7, "Cubic Out"}},
          {tweeny::easing::enumerated::cubicInOut, {8, "Cubic InOut"}},
          {tweeny::easing::enumerated::quarticIn, {9, "Quartic In"}},
          {tweeny::easing::enumerated::quarticOut, {10, "Quartic Out"}},
          {tweeny::easing::enumerated::quarticInOut, {11, "Quartic InOut"}},
          {tweeny::easing::enumerated::quinticIn, {12, "Quintic In"}},
          {tweeny::easing::enumerated::quinticOut, {13, "Quintic Out"}},
          {tweeny::easing::enumerated::quinticInOut, {14, "Quintic InOut"}},
          {tweeny::easing::enumerated::sinusoidalIn, {15, "Sinusoidal In"}},
          {tweeny::easing::enumerated::sinusoidalOut, {16, "Sinusoidal Out"}},
          {tweeny::easing::enumerated::sinusoidalInOut,
           {17, "Sinusoidal InOut"}},
          {tweeny::easing::enumerated::exponentialIn, {18, "Exponential In"}},
          {tweeny::easing::enumerated::exponentialOut, {19, "Exponential Out"}},
          {tweeny::easing::enumerated::exponentialInOut,
           {20, "Exponential InOut"}},
          {tweeny::easing::enumerated::circularIn, {21, "Circular In"}},
          {tweeny::easing::enumerated::circularOut, {22, "Circular Out"}},
          {tweeny::easing::enumerated::circularInOut, {23, "Circular InOut"}},
          {tweeny::easing::enumerated::bounceIn, {24, "Bounce In"}},
          {tweeny::easing::enumerated::bounceOut, {25, "Bounce Out"}},
          {tweeny::easing::enumerated::bounceInOut, {26, "Bounce InOut"}},
          {tweeny::easing::enumerated::elasticIn, {27, "Elastic In"}},
          {tweeny::easing::enumerated::elasticOut, {28, "Elastic Out"}},
          {tweeny::easing::enumerated::elasticInOut, {29, "Elastic InOut"}},
          {tweeny::easing::enumerated::backIn, {30, "Back In"}},
          {tweeny::easing::enumerated::backOut, {31, "Back Out"}},
          {tweeny::easing::enumerated::backInOut, {32, "Back InOut"}}};

  static const std::string combo_item_string = [] {
    std::string items;
    for (const auto &[mode, item] : ease_function_to_combo_item) {
      items += item.second + '\0';
    }
    return items;
  }();

  auto [selected_item_index, item_label] =
      ease_function_to_combo_item.at(config.ease_function);

  if (ImGui::Combo(std::string("##" + item_label).c_str(), &selected_item_index,
                   combo_item_string.c_str())) {
    for (const auto &[ease_function, combo_item] :
         ease_function_to_combo_item) {
      if (combo_item.first == selected_item_index) {
        config.ease_function = ease_function;
        break;
      }
    }
  }

  ImGui::PopID();
}

} // namespace pc::gui
