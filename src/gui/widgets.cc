#include "widgets.h"
#include "../logger.h"
#include "../string_utils.h"
#include "catpuccin.h"
#include "drag_slider.h"
#include <array>
#include <imgui.h>
#include <imgui_internal.h>
#include <tweeny/easing.h>

namespace pc::gui {

unsigned int _parameter_index;
Mode _current_mode;
std::string_view _modeline_input;

void begin_gui_helpers(
    const Mode current_mode,
    const std::array<char, modeline_buffer_size> &modeline_input) {
  _parameter_index = 0;
  _current_mode = current_mode;
  if (current_mode == Mode::Find) {
    // find the first element equal to null terminator to get the actual input
    // length
    constexpr char default_initialised_char = {};
    const auto modeline_input_end = std::find(
        modeline_input.begin(), modeline_input.end(), default_initialised_char);
    const std::size_t input_length =
        std::distance(modeline_input.begin(), modeline_input_end);
    _modeline_input = std::string_view(modeline_input.data(), input_length);
  }
}

void init_parameter_styles() {
  ImGui::StyleColorsDark();
  auto &style = ImGui::GetStyle();
  style.WindowPadding = {0, 10};
  style.FramePadding = {2, 3.5};
  style.ItemInnerSpacing = {5, 4};
  style.ItemSpacing = {1, 0};

  constexpr auto transparent = ImColor(0, 0, 0, 0);

  using namespace catpuccin;
  using namespace catpuccin::imgui;

  style.Colors[ImGuiCol_WindowBg] = rgba<ImVec4>(mocha_crust, 0.9f);
  style.Colors[ImGuiCol_Border] = mocha_surface2;

  style.Colors[ImGuiCol_Text] = mocha_text;
  style.Colors[ImGuiCol_TitleBg] = mocha_crust;
  style.Colors[ImGuiCol_TitleBgActive] = mocha_base;

  style.Colors[ImGuiCol_Header] = mocha_base;
  style.Colors[ImGuiCol_HeaderHovered] = mocha_surface;
  style.Colors[ImGuiCol_HeaderActive] = mocha_surface1;

  style.Colors[ImGuiCol_Tab] = mocha_surface;
  style.Colors[ImGuiCol_TabActive] = mocha_surface1;
  style.Colors[ImGuiCol_TabUnfocusedActive] = macchiato_surface1;
  style.Colors[ImGuiCol_TabHovered] = mocha_surface2;

  // TableHeaderBg is used for Parameter labels
  // style.Colors[ImGuiCol_TableHeaderBg] = mocha_crust;
  style.Colors[ImGuiCol_TableHeaderBg] = transparent;

  style.Colors[ImGuiCol_FrameBg] = mocha_base;
  style.Colors[ImGuiCol_FrameBgHovered] = mocha_surface;
  style.Colors[ImGuiCol_FrameBgActive] = mocha_surface1;

  style.Colors[ImGuiCol_Button] = mocha_base;
  style.Colors[ImGuiCol_ButtonHovered] = mocha_surface1;
  style.Colors[ImGuiCol_ButtonActive] = mocha_surface;

  style.Colors[ImGuiCol_CheckMark] = mocha_overlay;

  style.Colors[ImGuiCol_NavHighlight] = mocha_blue;

  style.Colors[ImGuiCol_ScrollbarGrab] = mocha_surface;
  style.Colors[ImGuiCol_ScrollbarGrabHovered] = mocha_surface1;
  style.Colors[ImGuiCol_ScrollbarGrabActive] = macchiato_surface2;

  style.Colors[ImGuiCol_PopupBg] = mocha_mantle;
};

std::string format_label(std::string_view label) {
  return strings::sentence_case(strings::last_element(label));
};

template <typename T>
bool slider(std::string_view parameter_id, T &value, T min, T max,
            T reset_value, bool is_disabled, std::string_view label) {

  ImGui::PushID(parameter_id.data());

  if (is_disabled)
    ImGui::BeginDisabled();

  const auto state = parameter_states[parameter_id];
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
    if (!learning_parameter)
      new_state = recording_result;
  }

  std::string label_text;
  if (label.empty()) {
    label_text = format_label(parameter_id).data();
  } else {
    label_text = label;
  }

  bool updated = false;

  if (state != ParameterState::Bound) {
    if constexpr (std::same_as<T, float>) {
      updated = ImGui::DragFloat(parameter_id.data(), &value, 0.01f, min, max,
                                 "%.5g");
    } else if constexpr (std::same_as<T, int>) {
      updated = ImGui::DragInt(parameter_id.data(), &value, 1.0f, min, max);
    }
  } else {
    // if the slider is bound, draw a range slider to set the min and max values
    ImGui::SetNextItemWidth(-1);
    auto &binding = parameter_bindings.at(parameter_id);
    auto old_binding = binding;
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
  ImGui::SameLine();
  if (state == ParameterState::Bound)
    ImGui::BeginDisabled();
  if (ImGui::Button("·", {15, 18})) {
    value = reset_value;
    updated = true;
  }
  if (state == ParameterState::Bound)
    ImGui::EndDisabled();

  ImGui::PopStyleColor(state != ParameterState::Unbound ? 5 : 0);

  if (is_disabled)
    ImGui::EndDisabled();

  parameter_states[parameter_id] = new_state;

  if (unbind_current)
    unbind_parameter(parameter_id);

  ImGui::PopID();

  return updated;
}

template bool slider(std::string_view parameter_id, float &value, float min,
                     float max, float reset_value, bool is_disabled,
                     std::string_view label);

template bool slider(std::string_view parameter_id, int &value, int min,
                     int max, int reset_value, bool is_disabled,
                     std::string_view label);

bool bool_param(std::string_view group_id, std::string_view parameter_id,
                bool &value, const bool &reset_value) {

  using namespace ImGui;

  ImGuiContext &g = *GImGui;
  auto *window = ImGui::GetCurrentWindow();

  const auto original_value = value;

  const auto parameter_label = fmt::format("{}.{}", group_id, parameter_id);
  const auto imgui_parameter_id =
      parameter_label + "." + std::to_string(_parameter_index++);
  const auto formatted_label =
      strings::sentence_case(strings::last_element(parameter_label));

  ImGui::PushID(imgui_parameter_id.c_str());

  const auto label_width = std::min(window->Size.x / 3.0f, 130.0f);

  const auto frame_start = window->DC.CursorPos;
  const auto text_width = label_width - 1;
  const auto frame_end = frame_start +
                         ImVec2{text_width, GetTextLineHeightWithSpacing()} +
                         g.Style.ItemInnerSpacing + ImVec2{0, 3};
  const auto frame_width = frame_end.x - frame_start.x;
  const ImRect frame_bb(frame_start, frame_end);
  const ImRect text_bb(ImVec2{frame_end.x, frame_start.y} -
                           ImVec2{text_width, 0} + g.Style.ItemInnerSpacing,
                       frame_end - g.Style.ItemInnerSpacing * 0.5f);

  ImGui::ItemAdd(text_bb,
                 GetID(fmt::format("{}.{}", parameter_id, "label").c_str()));
  if (ImGui::IsItemHovered()) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{11, 6});
    ImGui::SetTooltip("%s", parameter_id.data());
    ImGui::PopStyleVar();
  }

  Dummy({text_width + g.Style.ItemInnerSpacing.x, 0});
  RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_TableHeaderBg));

  RenderTextClipped(text_bb.Min, text_bb.Max, formatted_label.data(),
	  formatted_label.data() + formatted_label.size(), NULL, ImVec2(1.0f, 0.5f));
  SameLine();
  Dummy(g.Style.ItemSpacing);
  SameLine();

  Checkbox(fmt::format("##{}", parameter_id).c_str(), &value);

  Dummy(g.Style.FramePadding);

  ImGui::PopID();

  return value != original_value;
}

bool string_param(std::string_view group_id, std::string_view parameter_id,
                  std::string &value, const std::string &reset_value) {

  using namespace ImGui;

  ImGuiContext &g = *GImGui;
  auto *window = ImGui::GetCurrentWindow();

  const auto original_value = value;

  const auto parameter_label = fmt::format("{}.{}", group_id, parameter_id);
  const auto imgui_parameter_id =
      parameter_label + "." + std::to_string(_parameter_index++);
  const auto formatted_label =
      strings::sentence_case(strings::last_element(parameter_label));

  ImGui::PushID(imgui_parameter_id.c_str());

  const auto label_width = std::min(window->Size.x / 3.0f, 130.0f);

  const auto frame_start = window->DC.CursorPos;
  const auto text_width = label_width - 1;
  const auto frame_end = frame_start +
                         ImVec2{text_width, GetTextLineHeightWithSpacing()} +
                         g.Style.ItemInnerSpacing + ImVec2{0, 3};
  const auto frame_width = frame_end.x - frame_start.x;
  const ImRect frame_bb(frame_start, frame_end);
  const ImRect text_bb(ImVec2{frame_end.x, frame_start.y} -
                           ImVec2{text_width, 0} + g.Style.ItemInnerSpacing,
                       frame_end - g.Style.ItemInnerSpacing * 0.5f);

  ImGui::ItemAdd(text_bb,
                 GetID(fmt::format("{}.{}", parameter_id, "label").c_str()));
  if (ImGui::IsItemHovered()) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{11, 6});
    ImGui::SetTooltip("%s", parameter_id.data());
    ImGui::PopStyleVar();
  }

  Dummy({text_width + g.Style.ItemInnerSpacing.x, 0});
  RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_TableHeaderBg));

  RenderTextClipped(text_bb.Min, text_bb.Max, formatted_label.data(),
	  formatted_label.data() + formatted_label.size(), NULL, ImVec2(1.0f, 0.5f));
  SameLine();
  Dummy(g.Style.ItemSpacing);
  SameLine();

  // TODO investigate arbitrary string length
  InputText(fmt::format("##{}", parameter_id).c_str(), value.data(), 32);

  Dummy(g.Style.FramePadding);

  ImGui::PopID();

  return value != original_value;
}

template <pc::types::ScalarType T>
bool scalar_param(std::string_view group_id, std::string_view parameter_id,
                  T &value, T min, T max, T reset_value, bool disabled,
                  std::string_view label) {

  auto original_value = value;

  auto imgui_parameter_id =
      fmt::format("{}.{}", parameter_id, _parameter_index++);
  ImGui::PushID(imgui_parameter_id.c_str());

  if constexpr (std::floating_point<T>) {
    pc::gui::DragFloat(parameter_id, &value, 0.0001f, min, max, reset_value);
  } else if constexpr (std::same_as<T, int>) {
    pc::gui::DragInt(parameter_id, &value, 1, min, max, reset_value);
  } else if constexpr (std::same_as<T, short>) {
    pc::logger->debug("right sport here");
  }

  ImGui::PopID();

  return value != original_value;
}

template bool scalar_param(std::string_view group_id,
                           std::string_view parameter_id, float &value,
                           float min, float max, float reset_value,
                           bool disabled, std::string_view label);

template bool scalar_param(std::string_view group_id,
                           std::string_view parameter_id, int &value, int min,
                           int max, int reset_value, bool disabled,
                           std::string_view label);

template <typename T>
bool vector_param(std::string_view group_id, std::string_view parameter_id,
                  T &vec, typename T::vector_type min,
                  typename T::vector_type max, T reset_values,
                  std::array<bool, types::VectorSize<T>::value> disabled,
                  std::array<std::string, types::VectorSize<T>::value> labels) {

  constexpr auto vector_size = types::VectorSize<T>::value;

  auto original_vec = vec;

  auto imgui_parameter_id =
      fmt::format("{}.{}", parameter_id, _parameter_index++);
  ImGui::PushID(imgui_parameter_id.c_str());

  const float drag_range = max - min;
  const float drag_speed = drag_range / 2000.0f;

  using ElementT = typename T::vector_type;

  if constexpr (std::same_as<ElementT, float>) {
    if constexpr (vector_size == 2) {
      pc::gui::DragFloat2(parameter_id, vec.data(), drag_speed, min, max,
                          reset_values);
    } else if constexpr (vector_size == 3) {
      pc::gui::DragFloat3(parameter_id, vec.data(), drag_speed, min, max,
                          reset_values);
    } else if constexpr (vector_size == 4) {
      pc::logger->warn("implement Float4 parameters");
      // pc::gui::DragFloat4(parameter_label, vec.data(), drag_speed, min, max);
    }
  } else if constexpr (std::same_as<ElementT, int>) {
    if constexpr (vector_size == 2) {
      pc::gui::DragInt2(parameter_id, vec.data(), 1, min, max, reset_values);
    } else if constexpr (vector_size == 3) {
      pc::gui::DragInt3(parameter_id, vec.data(), 1, min, max, reset_values);
    } else if constexpr (vector_size == 4) {
      pc::logger->warn("implement Int4 parameters");
      // pc::gui::DragInt4(parameter_label, vec.data(), drag_speed, min, max,
      // reset_values);
    }
  } else if constexpr (std::same_as<ElementT, short>) {
    if constexpr (vector_size == 2) {
      pc::gui::DragShort2(parameter_id, vec.data(), 1, min, max, reset_values);
    }
  }

  ImGui::PopID();

  return vec != original_vec;
}

template bool vector_param(std::string_view group_id,
                           std::string_view parameter_id, Int2 &vec, int min,
                           int max, Int2 reset_values,
                           std::array<bool, 2> disabled,
                           std::array<std::string, 2> labels);

template bool vector_param(std::string_view group_id,
                           std::string_view parameter_id, Int3 &vec, int min,
                           int max, Int3 reset_values,
                           std::array<bool, 3> disabled,
                           std::array<std::string, 3> labels);

template bool vector_param(std::string_view group_id,
                           std::string_view parameter_id, MinMax<int> &vec,
                           int min, int max, MinMax<int> reset_values,
                           std::array<bool, 2> disabled,
                           std::array<std::string, 2> labels);

template bool vector_param(std::string_view group_id,
                           std::string_view parameter_id, Float2 &vec,
                           float min, float max, Float2 reset_values,
                           std::array<bool, 2> disabled,
                           std::array<std::string, 2> labels);

template bool vector_param(std::string_view group_id,
                           std::string_view parameter_id, Float3 &vec,
                           float min, float max, Float3 reset_values,
                           std::array<bool, 3> disabled,
                           std::array<std::string, 3> labels);

template bool vector_param(std::string_view group_id,
                           std::string_view parameter_id, Float4 &vec,
                           float min, float max, Float4 reset_values,
                           std::array<bool, 4> disabled,
                           std::array<std::string, 4> labels);

template bool vector_param(std::string_view group_id,
                           std::string_view parameter_id, MinMax<float> &vec,
                           float min, float max, MinMax<float> reset_values,
                           std::array<bool, 2> disabled,
                           std::array<std::string, 2> labels);

template bool vector_param(std::string_view group_id,
                           std::string_view parameter_id, MinMax<short> &vec,
                           short min, short max, MinMax<short> reset_values,
                           std::array<bool, 2> disabled,
                           std::array<std::string, 2> labels);

template bool vector_param(std::string_view group_id,
                           std::string_view parameter_id, MinMax<short> &vec,
                           short min, short max, short reset_value,
                           std::array<bool, 2> disabled,
                           std::array<std::string, 2> labels);

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

bool draw_parameter(std::string_view structure_name,
                    std::string_view parameter_id) {

  static const std::array<std::string, 2> ignored_suffixes = { "unfolded", ".show_window"};
  if (pc::strings::ends_with_any(parameter_id, ignored_suffixes.begin(),
                                 ignored_suffixes.end())) {
    return false;
  }

  auto param = parameter_bindings.at(parameter_id);
  auto original_param = param;

  bool updated = std::visit(
      [parameter_id, structure_name, &param](auto &&ref) {
        using T = std::decay_t<decltype(ref.get())>;
        // bools are drawn as check boxes
        if constexpr (std::same_as<T, bool>) {

          return pc::gui::bool_param(structure_name, parameter_id, ref.get(),
                                     std::get<bool>(param.default_value));

        } else if constexpr (std::same_as<T, std::string>) {
          return pc::gui::string_param(
              structure_name, parameter_id, ref.get(),
              std::get<std::string>(param.default_value));
        }
        // floats and ints are drawn individually
        else if constexpr (std::floating_point<T> || std::integral<T>) {
          return pc::gui::scalar_param(
              structure_name, parameter_id, ref.get(), std::get<T>(param.min),
              std::get<T>(param.max), std::get<T>(param.default_value));
        }
        // vector multi-parameters are drawn together
        else if constexpr (is_float_vector_t<T> || is_int_vector_t<T> ||
                           is_short_vector_t<T>) {
          return pc::gui::vector_param(
              structure_name, parameter_id, ref.get(),
              std::get<typename T::vector_type>(param.min),
              std::get<typename T::vector_type>(param.max),
              std::get<T>(param.default_value));
        }
      },
      param.value);

  if (updated) {
    for (const auto &cb : param.update_callbacks) { cb(original_param, param); }
  }
  return updated;
}

bool draw_parameters(std::string_view structure_name, const ParameterMap &map,
                     const std::string &map_prefix) {
  auto changed = false;
  ImGui::PushID(std::to_string(_parameter_index++).c_str());
  for (const auto &entry : map) {
    if (std::holds_alternative<std::string>(entry)) {
      auto entry_str = std::get<std::string>(entry);
      // don't allow editable string ids through automated parameter drawing
      if (std::string_view(entry_str).contains(".id")) { continue; }
      if (std::string_view(entry_str).contains(".active")) { continue; }
      ImGui::PushID(std::to_string(_parameter_index++).c_str());
      changed |= draw_parameter(structure_name, entry_str);
      ImGui::Dummy({0, 1});
      ImGui::PopID();
    } else {
      // this entry holds a nested map,
      // draw nested params inside a collapsing header
      const auto &nested_entry = std::get<NestedParameterMap>(entry);
      const auto &entry_name = nested_entry.first;
      const auto &nested_map = nested_entry.second;
      const auto header_text = pc::strings::sentence_case(entry_name);

      // grab the 'unfolded' entry to determine whether the collapsing header is
      // open or closed (if that unfolded param exists in the structure)

      std::optional<BoolReference> unfolded;

      const auto unfolded_entry_id = std::string(structure_name) + "." +
                                     map_prefix + entry_name + ".unfolded";
      auto unfolded_it = parameter_bindings.find(unfolded_entry_id);

      if (unfolded_it != parameter_bindings.end()) {
        const auto &unfolded_param = parameter_bindings.at(unfolded_entry_id);
        unfolded = std::get<BoolReference>(unfolded_param.value);
      }

      ImGui::PushID(std::to_string(_parameter_index++).c_str());

      if (unfolded.has_value()) {
        ImGui::SetNextItemOpen(unfolded.value().get());
      }

      if (ImGui::CollapsingHeader(header_text.data())) {
        ImGui::Dummy({0, 4});
        // call draw recursively inside the header this time
        changed |= draw_parameters(structure_name, nested_map,
                                   map_prefix + entry_name + ".");
        ImGui::Dummy({0, 1});
        if (unfolded.has_value()) {
          unfolded.value().get() = true;
        }
      } else {
        if (unfolded.has_value()) {
          unfolded.value().get() = false;
        }
        ImGui::Dummy({0, 0});
      }

      ImGui::PopID();
    }
  }
  ImGui::PopID();;
  return changed;
}

bool draw_parameters(std::string_view structure_id) {
  // TODO std::string creation every frame
  auto id = std::string{structure_id};
  if (!struct_parameters.contains(id)) {
    pc::logger->warn("Unable to find parameter '{}' for drawing", id);
    return false;
  }
  return draw_parameters(structure_id, struct_parameters.at(id));
}

bool draw_parameters(unsigned long int structure_id) {
  return draw_parameters(std::to_string(structure_id));
}

bool draw_icon_button(std::string_view icon, bool small, ImVec4 default_color,
                      ImVec4 hover_color) {
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  if (window->SkipItems) return false;

  ImGui::PushFont(icon_font.get());

  ImGuiContext &g = *ImGui::GetCurrentContext();
  const ImGuiID id = window->GetID(pc::gui::_parameter_index++);
  const ImVec2 pos = window->DC.CursorPos;
  const ImVec2 text_size = ImGui::CalcTextSize(icon.data());
  const ImVec2 padding = g.Style.FramePadding;
  const ImVec2 button_size = ImGui::CalcItemSize(
      {0, 0}, text_size.x + padding.x * 2.0f, text_size.y + padding.y * 2.0f);
  const ImRect bb(pos, pos + button_size);
  ImGui::ItemSize(bb, padding.y);
  if (!ImGui::ItemAdd(bb, id)) {
    ImGui::PopFont();
    return false;
  }

  bool hovered, held;
  bool pressed = ImGui::ButtonBehavior(bb, id, &hovered, &held);

  auto text_color = hovered ? hover_color : default_color;
  ImVec2 text_pos(pos.x + (button_size.x - text_size.x) * 0.5f,
                  pos.y + (button_size.y - text_size.y) * 0.5f);
  window->DrawList->AddText(text_pos, ImGui::GetColorU32(text_color),
                            icon.data());

  ImGui::PopFont();

  return pressed;
}

} // namespace pc::gui
