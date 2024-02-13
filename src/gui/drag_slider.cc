#include "drag_slider.h"
#include "catpuccin.h"
#include "../parameters.h"
#include "../string_utils.h"
#include <array>
#include <cstring>
#include <imgui_internal.h>
#include <string>

namespace pc::gui {

using namespace ImGui;

static const float DRAG_MOUSE_THRESHOLD_FACTOR =
    0.50f; // Multiplier for the default value of io.MouseDragThreshold to make
           // DragFloat/DragInt react faster to mouse drags.:

static const ImGuiDataTypeInfo GDataTypeInfo[] = {
    {sizeof(char), "S8", "%d", "%d"}, // ImGuiDataType_S8
    {sizeof(unsigned char), "U8", "%u", "%u"},
    {sizeof(short), "S16", "%d", "%d"}, // ImGuiDataType_S16
    {sizeof(unsigned short), "U16", "%u", "%u"},
    {sizeof(int), "S32", "%d", "%d"}, // ImGuiDataType_S32
    {sizeof(unsigned int), "U32", "%u", "%u"},
#ifdef _MSC_VER
    {sizeof(ImS64), "S64", "%I64d", "%I64d"}, // ImGuiDataType_S64
    {sizeof(ImU64), "U64", "%I64u", "%I64u"},
#else
    {sizeof(ImS64), "S64", "%lld", "%lld"}, // ImGuiDataType_S64
    {sizeof(ImU64), "U64", "%llu", "%llu"},
#endif
    {sizeof(float), "float", "%.3f",
     "%f"}, // ImGuiDataType_Float (float are promoted to double in va_arg)
    {sizeof(double), "double", "%f", "%lf"}, // ImGuiDataType_Double
};

inline float label_width(auto& window_size) {
  return std::min(window_size / 3.0f, 130.0f);
}

bool DragScalar(int component_index, int component_count,
		std::string_view parameter_id, ImGuiDataType data_type, void *p_data,
		float v_speed, const void *p_min, const void *p_max,
                const void *p_reset, const char *format = "%.2f",
                ImGuiSliderFlags flags = 0) {
  ImGuiWindow *window = GetCurrentWindow();
  if (window->SkipItems)
    return false;

  //parameter learning

  auto &param_state = parameter_states[parameter_id];
  auto new_param_state = param_state;
  auto unbind_param = false;

  if (param_state == ParameterState::Bound) {
    ImGui::PushStyleColor(ImGuiCol_Text, catpuccin::mocha_lavender);

  } else if (param_state == ParameterState::Learning) {
    ImGui::PushStyleColor(ImGuiCol_Text, catpuccin::mocha_red);

    if (!pc::gui::learning_parameter) {
      // if we *were* recording a param and we're not anymore,
      // set its state to the result of the recording
      new_param_state = pc::gui::recording_result;
    }
  }

  // draw the control

  ImGuiContext &g = *GImGui;
  const ImGuiStyle &style = g.Style;
  const auto end_distance = ImGui::GetWindowContentRegionMax().x -
			    ImGui::GetCursorPosX() -
			    g.Style.ItemInnerSpacing.x;
  const auto element_width =
      end_distance / static_cast<float>(component_count - component_index);

  ImGui::SetNextItemWidth(element_width);
  auto value_changed =
      ImGui::DragScalar(fmt::format("##{}", parameter_id).c_str(), data_type,
			p_data, v_speed, p_min, p_max);

  // mouse handling

  if (ImGui::IsItemHovered()) {

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
      // Alt + Left click resets the value to default
      if (ImGui::IsKeyDown(ImGuiKey_LeftAlt)) {
	std::memcpy(p_data, p_reset, GDataTypeInfo[data_type].Size);
      }
    }
    // Right click starts parameter learn
    else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
      if (!learning_parameter) {
	// if we were not learning, set it to learn
	new_param_state = ParameterState::Learning;
	pc::gui::learning_parameter = true;
	auto f = reinterpret_cast<float*>(p_data);
	auto min = reinterpret_cast<const float*>(p_min);
	auto max = reinterpret_cast<const float*>(p_max);
	store_learning_parameter_info(parameter_id, *min, *max, *f);
      } else {
	// if we were learning and we right clicked, return the slider to an
	// unbound state
	new_param_state = ParameterState::Unbound;
	pc::gui::learning_parameter = false;
	unbind_param = true;
      }
    }
  }

  ImGui::PopStyleColor(param_state != ParameterState::Unbound ? 1 : 0);

  param_state = new_param_state;
  if (unbind_param) {
    unbind_parameter(parameter_id);
  }

  return value_changed;
}

bool DragScalarN(std::string_view parameter_id, ImGuiDataType data_type, void *p_data,
                 int components, float v_speed, const void *p_min,
                 const void *p_max, std::vector<const void *> p_reset, const char *format,
                 ImGuiSliderFlags flags) {
  ImGuiWindow *window = GetCurrentWindow();
  if (window->SkipItems)
    return false;

  ImGuiContext &g = *GImGui;
  bool value_changed = false;
  BeginGroup();
  PushID(parameter_id.data());

  const auto scalar_name = strings::last_element(parameter_id);
  const auto formatted_label = strings::sentence_case(scalar_name);

  const auto find_mode = _current_mode == Mode::Find;
  const auto navigate_match = _current_mode == Mode::NavigateMatch;

  size_t label_highlight_start = 0;
  size_t label_highlight_end = 0;

  // check for a match with our labels if we're in find mode
  if (find_mode || navigate_match) {
    // TODO matching on the name and the label to ignore capitalisation
    // const auto element_match = scalar_name.find(_modeline_input);
    const auto label_match = formatted_label.find(_modeline_input);

    // if (element_match != std::string::npos) {
    //   pc::logger->debug("found_parameter element_match with {}", parameter_id);
    // }

    if (_modeline_input.size() != 0 && label_match != std::string::npos) {
      label_highlight_start = label_match;
      label_highlight_end = label_match + _modeline_input.length();
      pc::logger->debug(formatted_label);
    }
  }

  const auto frame_start = window->DC.CursorPos;
  const auto text_width = label_width(window->Size.x) - 1;
  const auto frame_end = frame_start +
			 ImVec2{text_width, GetTextLineHeightWithSpacing()} +
			 g.Style.ItemInnerSpacing + ImVec2{0, 3};
  const auto frame_width = frame_end.x - frame_start.x;
  const ImRect frame_bb(frame_start, frame_end);
  const auto frame_size = frame_bb.Max - frame_bb.Min;
  const ImRect text_bb(ImVec2{frame_end.x, frame_start.y} -
			   ImVec2{text_width, 0} + g.Style.ItemInnerSpacing,
		       frame_end - g.Style.ItemInnerSpacing * 0.5f);

  ItemAdd(text_bb, GetID(fmt::format("{}.{}", parameter_id, "l").c_str()));
  if (IsItemHovered()) {
    PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{11, 6});
    SetTooltip("%s", parameter_id.data());
    PopStyleVar();
  }

  Dummy({text_width + g.Style.ItemInnerSpacing.x, 0});

  RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_TableHeaderBg));

  const char *text_start = formatted_label.c_str();
  const char *text_end = text_start + formatted_label.size();

  if (label_highlight_end == 0) {
    // Render text normally
    const auto render_disabled = find_mode && !_modeline_input.empty();
    if (render_disabled)
      ImGui::BeginDisabled();
    RenderTextClipped(text_bb.Min, text_bb.Max, text_start, text_end, NULL,
                      ImVec2(1.0f, 0.5f));
    if (render_disabled)
      ImGui::EndDisabled();
  } else {
    ImGui::PushStyleColor(ImGuiCol_Text, catpuccin::mocha_maroon);

    ImVec2 text_pos = text_bb.Max;

    // Render text before highlight
    ImGui::BeginDisabled();
    RenderTextClipped(text_bb.Min, text_bb.Max, text_start,
                      text_start + label_highlight_start, NULL,
                      ImVec2(1.0f, 0.5f));
    ImGui::EndDisabled();

    text_pos.x -=
        ImGui::CalcTextSize(text_start, text_start + label_highlight_start).x;

    // Render highlighted text
    RenderTextClipped(
        text_bb.Min, text_bb.Max, text_start + label_highlight_start,
        text_start + label_highlight_end, NULL, ImVec2(1.0f, 0.5f));

    text_pos.x += ImGui::CalcTextSize(text_start + label_highlight_start,
                                      text_start + label_highlight_end)
                      .x;

    // Render text after highlight
    ImGui::BeginDisabled();
    RenderTextClipped(text_bb.Min, text_bb.Max, text_start + label_highlight_end,
		      text_end, NULL, ImVec2(1.0f, 0.5f));
    ImGui::EndDisabled();

    ImGui::PopStyleColor();
  }

  SameLine();
  Dummy(g.Style.ItemSpacing);
  SameLine();

  // draw the scalar or the vector element
  size_t type_size = GDataTypeInfo[data_type].Size;
  for (int i = 0; i < components; i++) {
    PushID(i);
    if (i > 0) {
      SameLine(0, g.Style.ItemInnerSpacing.x);
    }
    value_changed |=
        DragScalar(i, components, parameter_id, data_type, p_data, v_speed,
                   p_min, p_max, p_reset[i], format, flags);
    PopID();
    p_data = (void *)((char *)p_data + type_size);
  }
  PopID();
  Dummy(g.Style.FramePadding);
  EndGroup();
  return value_changed;
}

bool DragFloat(std::string_view label, float *v, float v_speed, float v_min,
               float v_max, float v_reset, const char *format,
               ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_Float, v, 1, v_speed, &v_min, &v_max,
                     {&v_reset}, format, flags);
}

bool DragFloat2(std::string_view label, float v[2], float v_speed, float v_min,
                float v_max, Float2 v_reset, const char *format,
                ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_Float, v, 2, v_speed, &v_min, &v_max,
                     {&v_reset[0], &v_reset[1]}, format, flags);
}

bool DragFloat2(std::string_view label, float v[2], float v_speed, float v_min,
                float v_max, pc::types::MinMax<float> v_reset, const char *format,
                ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_Float, v, 2, v_speed, &v_min, &v_max,
                     {&v_reset[0], &v_reset[1]}, format, flags);
}

bool DragFloat3(std::string_view label, float v[3], float v_speed, float v_min,
                float v_max, Float3 v_reset, const char *format,
                ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_Float, v, 3, v_speed, &v_min, &v_max,
                     {&v_reset[0], &v_reset[1], &v_reset[2]}, format, flags);
}

bool DragFloat4(std::string_view label, float v[4], float v_speed, float v_min,
                float v_max, Float4 v_reset, const char *format,
                ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_Float, v, 4, v_speed, &v_min, &v_max,
                     {&v_reset[0], &v_reset[1], &v_reset[2], &v_reset[3]},
                     format, flags);
}

bool DragInt(std::string_view label, int *v, int v_speed, int v_min, int v_max,
             int v_reset, const char *format, ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_S32, v, 1, v_speed, &v_min, &v_max,
                     {&v_reset}, format, flags);
}

bool DragInt2(std::string_view label, int v[2], int v_speed, int v_min,
              int v_max, Int2 v_reset, const char *format,
              ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_S32, v, 2, v_speed, &v_min, &v_max,
                     {&v_reset[0], &v_reset[1]}, format, flags);
}

bool DragShort(std::string_view label, short *v, short v_speed, short v_min, short v_max,
             short v_reset, const char *format, ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_S16, v, 1, v_speed, &v_min, &v_max,
                     {&v_reset}, format, flags);
}

bool DragShort2(std::string_view label, short v[2], short v_speed, short v_min,
		short v_max, types::Short2 v_reset, const char *format,
		ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_S16, v, 2, v_speed, &v_min, &v_max,
		     {&v_reset[0], &v_reset[1]}, format, flags);
}

bool DragShort2(std::string_view label, short v[2], short v_speed, short v_min,
              short v_max, pc::types::MinMax<short> v_reset, const char *format,
              ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_S16, v, 2, v_speed, &v_min, &v_max,
                     {&v_reset[0], &v_reset[1]}, format, flags);
}

bool DragInt2(std::string_view label, int v[2], int v_speed, int v_min,
              int v_max, pc::types::MinMax<int> v_reset, const char *format,
              ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_S32, v, 2, v_speed, &v_min, &v_max,
                     {&v_reset[0], &v_reset[1]}, format, flags);
}

bool DragInt3(std::string_view label, int v[3], int v_speed, int v_min,
              int v_max, Int3 v_reset, const char *format,
              ImGuiSliderFlags flags) {
  return DragScalarN(label, ImGuiDataType_S32, v, 3, v_speed, &v_min, &v_max,
                     {&v_reset[0], &v_reset[1], &v_reset[2]}, format, flags);
}

} // namespace pc::gui
