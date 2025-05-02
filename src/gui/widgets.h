#pragma once

#include "../fonts/IconsFontAwesome6.h"
#include "../math.h"
#include "../modes.h"
#include "../parameters.h"
#include "../structs.h"
#include "../tween/tween_config.gen.h"
#include "catpuccin.h"
#include <array>
#include <atomic>
#include <functional>
#include <imgui.h>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace pc::gui {

using pc::types::Float;
using pc::types::Float2;
using pc::types::Float3;
using pc::types::Float4;
using pc::types::Int;
using pc::types::Int2;
using pc::types::Int3;

using namespace pc::parameters;

/* Must be called at the start of the application */
void init_parameter_styles();

/* Must be called at the start of each imgui frame */
void begin_gui_helpers(const Mode current_mode,
		       const std::array<char, modeline_buffer_size>& modeline_input);

/* Used for generating unique ids for imgui parameters */
extern unsigned int _parameter_index;

/* these are updated at the start of each frame */
extern Mode _current_mode;
extern std::string_view _modeline_input;

// a cache for our tooltips, so that parameter tooltips show friendly names
inline std::map<std::string, std::string> parameter_tooltips;
inline std::atomic_bool refresh_tooltips;
extern std::string compute_tooltip(const std::string_view parameter_id);
extern const char *get_tooltip_cstr(const std::string_view parameter_id);
extern void show_parameter_tooltip(const std::string_view parameter_id);

constexpr ImVec2 tooltip_padding{10, 10};

inline std::atomic_bool learning_parameter;
inline std::atomic<ParameterState> recording_result;
inline std::string learning_parameter_id;
inline std::optional<Parameter> learning_parameter_info;
inline std::mutex learning_parameter_mutex;

inline std::shared_ptr<ImFont> icon_font;
inline std::shared_ptr<ImFont> icon_font_small;

template <typename T>
void store_learning_parameter_info(std::string_view id, float min, float max,
                                    T &value) {
  std::lock_guard<std::mutex> lock(learning_parameter_mutex);
  learning_parameter_id = id;
  learning_parameter_info = Parameter(value, min, max);
}

inline std::pair<std::string, Parameter>
load_learning_parameter_info() {
  std::lock_guard<std::mutex> lock(learning_parameter_mutex);
  return {learning_parameter_id, learning_parameter_info.value()};
}

template <typename T>
bool slider(std::string_view parameter_id, T &value, T min, T max,
	    T reset_value, bool is_disabled = false,
	    std::string_view label = "");

template <typename T>
bool slider(std::string_view group_id, std::string_view parameter_id, T &value,
	    T min, T max, T reset_value, bool is_disabled = false) {
  auto nested_id = fmt::format("{}/{}", group_id, parameter_id);
  return slider(std::string_view(nested_id), value, min, max, reset_value,
		is_disabled);
}

template <pc::types::ScalarType T>
bool scalar_param(std::string_view group_id, std::string_view parameter_id,
		  T &value, T min = -10, T max = 10, T reset_value = 0,
		  bool disabled = false, std::string_view label = "");

template <typename T>
bool vector_param(
    std::string_view group_id, std::string_view parameter_id, T &vec,
    typename T::vector_type min, typename T::vector_type max, T reset_values,
    std::array<bool, types::VectorSize<T>::value> disabled = {},
    std::array<std::string, types::VectorSize<T>::value> labels = {});

template <typename T>
bool vector_param(
    std::string_view group_id, std::string_view parameter_id, T &vec,
    typename T::vector_type min, typename T::vector_type max,
    typename T::vector_type reset_value,
    std::array<bool, types::VectorSize<T>::value> disabled = {},
    std::array<std::string, types::VectorSize<T>::value> labels = {}) {

  pc::logger->warn("drawing vector param with unimplemented overload");

  return false;

  // constexpr auto vector_size = types::VectorSize<T>::value;
  // std::array<typename T::vector_type, vector_size> reset_values;

  // for (std::size_t i = 0; i < vector_size; ++i) {
  //   reset_values[i] = reset_value[i];
  // }

  // return vector_param(group_id, parameter_id, vec, min, max, reset_values,
  // 		      disabled, labels);
}

template <typename T>
bool vector_param(std::string_view group_id, std::string_view parameter_id,
                  T &vec, typename T::vector_type min,
                  typename T::vector_type max,
                  typename T::vector_type reset_value,
                  bool disabled_value) {

  constexpr auto vector_size = types::VectorSize<T>::value;

  std::array<typename T::vector_type, vector_size> reset_values;
  reset_values.fill(reset_value);

  std::array<bool, vector_size> disabled_array;
  disabled_array.fill(disabled_value);

  return vector_param(group_id, parameter_id, vec, min, max, reset_values,
                      disabled_array);
}

bool begin_tree_node(std::string_view name, bool &open);

void tween_config(std::string_view label,
                  pc::tween::TweenConfiguration &config);

bool draw_parameters(std::string_view structure_name, const ParameterMap &map,
		     const std::string &map_prefix = "");

bool draw_parameters(std::string_view structure_id);

bool draw_parameters(unsigned long int structure_id);

bool draw_icon_button(std::string_view icon, bool small, ImVec4 default_color,
                      ImVec4 hover_color);

bool draw_icon_tab_button(std::string_view icon_string,
                          ImGuiTabItemFlags flags = ImGuiTabItemFlags_None,
                          ImVec2 pos_offset = {0, 0});

void push_context_menu_styles();

void pop_context_menu_styles();

} // namespace pc::gui
