#pragma once

#include "../fonts/IconsFontAwesome6.h"
#include "../math.h"
#include "../parameters.h"
#include "../structs.h"
#include "../tween/tween_config.h"
#include "range_slider.h"
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

using pc::types::Float2;
using pc::types::Float3;
using pc::types::Float4;
using pc::types::Int2;
using pc::types::Int3;

/* Must be called at the start of each imgui frame */
void begin_gui_helpers();

/* Used for generating unique ids for imgui parameters */
extern unsigned int _parameter_index;

inline std::atomic_bool learning_parameter;
inline std::atomic<ParameterState> recording_result;
inline std::string learning_parameter_id;
inline std::optional<ParameterBinding> learning_parameter_info;
inline std::mutex learning_parameter_mutex;

template <typename T>
void store_learning_parameter_info(std::string_view id, float min, float max,
                                    T &value) {
  std::lock_guard<std::mutex> lock(learning_parameter_mutex);
  learning_parameter_id = id;
  learning_parameter_info = ParameterBinding(value, min, max);
}

inline std::pair<std::string, ParameterBinding>
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
  auto nested_id = fmt::format("{}.{}", group_id, parameter_id);
  return slider(std::string_view(nested_id), value, min, max, reset_value,
		is_disabled);
}

template <typename T>
bool vector_table(
    std::string_view group_id, std::string_view parameter_id, T &vec,
    typename T::vector_type min, typename T::vector_type max,
    std::array<typename T::vector_type, types::VectorSize<T>::value>
	reset_values,
    std::array<bool, types::VectorSize<T>::value> disabled = {},
    std::array<std::string, types::VectorSize<T>::value> labels = {});

template <typename T>
bool vector_table(
    std::string_view group_id, std::string_view parameter_id, T &vec,
    typename T::vector_type min, typename T::vector_type max, T reset_value,
    std::array<bool, types::VectorSize<T>::value> disabled = {},
    std::array<std::string, types::VectorSize<T>::value> labels = {}) {

  constexpr auto vector_size = types::VectorSize<T>::value;
  std::array<typename T::vector_type, vector_size> reset_values;

  for (std::size_t i = 0; i < vector_size; ++i) {
    reset_values[i] = reset_value[i];
  }

  return vector_table(group_id, parameter_id, vec, min, max, reset_values,
		      disabled, labels);
}

template <typename T>
bool vector_table(std::string_view group_id, std::string_view parameter_id,
                  T &vec, typename T::vector_type min,
                  typename T::vector_type max,
                  typename T::vector_type reset_value,
                  bool disabled_value = false) {

  constexpr auto vector_size = types::VectorSize<T>::value;

  std::array<typename T::vector_type, vector_size> reset_values;
  reset_values.fill(reset_value);

  std::array<bool, vector_size> disabled_array;
  disabled_array.fill(disabled_value);

  return vector_table(group_id, parameter_id, vec, min, max, reset_values,
                      disabled_array);
}

bool begin_tree_node(std::string_view name, bool &open);

void tween_config(std::string_view label,
                  pc::tween::TweenConfiguration &config);

} // namespace pc::gui