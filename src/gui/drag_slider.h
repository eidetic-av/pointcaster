#pragma once

#include "../structs.h"
#include "widgets.h"
#include <imgui.h>
#include <string_view>

namespace pc::gui {

using pc::types::Float2;
using pc::types::Float3;
using pc::types::Float4;
using pc::types::Int2;
using pc::types::Int3;
using pc::types::MinMax;

bool DragFloat(std::string_view label, float *v, float v_speed, float v_min,
               float v_max, float v_reset = 0.0f, const char *format = "%.5f",
               ImGuiSliderFlags flags = 0);

bool DragFloat2(std::string_view label, float v[2], float v_speed, float v_min,
                float v_max, Float2 v_reset = {}, const char *format = "%.2f",
                ImGuiSliderFlags flags = 0);

bool DragFloat2(std::string_view label, float v[2], float v_speed, float v_min,
                float v_max, MinMax<float> v_reset = {}, const char *format = "%.2f",
                ImGuiSliderFlags flags = 0);

bool DragFloat3(std::string_view label, float v[3], float v_speed, float v_min,
                float v_max, Float3 v_reset = {}, const char *format = "%.2f",
                ImGuiSliderFlags flags = 0);

bool DragFloat4(std::string_view label, float v[4], float v_speed, float v_min,
                float v_max, Float4 v_reset = {}, const char *format = "%.2f",
                ImGuiSliderFlags flags = 0);

bool DragInt(std::string_view label, int *v, int v_speed, int v_min, int v_max,
             int v_reset = 0, const char *format = "%d",
             ImGuiSliderFlags flags = 0);

bool DragInt2(std::string_view label, int v[2], int v_speed, int v_min,
              int v_max, Int2 v_reset = {}, const char *format = "%d",
              ImGuiSliderFlags flags = 0);

bool DragInt2(std::string_view label, int v[2], int v_speed, int v_min,
              int v_max, MinMax<int> v_reset = {}, const char *format = "%d",
              ImGuiSliderFlags flags = 0);


bool DragInt3(std::string_view label, int v[3], int v_speed, int v_min,
              int v_max, Int3 v_reset = {}, const char *format = "%d",
              ImGuiSliderFlags flags = 0);

  // TODO
bool DragInt4(std::string_view label, int v[4], int v_speed, int v_min,
              int v_max, int v_reset = 0, const char *format = "%d",
              ImGuiSliderFlags flags = 0);

bool DragShort(std::string_view label, short *v, short v_speed, short v_min,
	       short v_max, short v_reset = 0, const char *format = "%d",
	       ImGuiSliderFlags flags = 0);

bool DragShort2(std::string_view label, short v[2], short v_speed, short v_min,
		short v_max, types::Short2 v_reset = {},
		const char *format = "%d", ImGuiSliderFlags flags = 0);

bool DragShort2(std::string_view label, short v[2], short v_speed, short v_min,
		short v_max, MinMax<short> v_reset = {},
		const char *format = "%d", ImGuiSliderFlags flags = 0);

} // namespace pc::gui
