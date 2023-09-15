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

using pc::types::float2;
using pc::types::float3;
using pc::types::float4;
using pc::types::int2;
using pc::types::int3;

/* Must be called at the start of each imgui frame */
void begin_gui_helpers();

/* Used for generating unique ids for imgui parameters */
extern unsigned int _parameter_index;

template <typename T>
void draw_slider(std::string_view label_text, T *value, T min, T max,
                 T default_value = 0);

inline std::atomic_bool recording_parameter;
inline std::atomic<ParameterState> recording_result;
inline std::string recording_parameter_id;
inline std::optional<ParameterBinding> recording_parameter_info;
inline std::mutex recording_parameter_mutex;

template <typename T>
void store_recording_parameter_info(std::string_view id, float min, float max,
                                    T &value) {
  std::lock_guard<std::mutex> lock(recording_parameter_mutex);
  recording_parameter_id = id;
  recording_parameter_info = ParameterBinding(value, min, max);
}

inline std::pair<std::string, ParameterBinding>
load_recording_parameter_info() {
  std::lock_guard<std::mutex> lock(recording_parameter_mutex);
  return {recording_parameter_id, recording_parameter_info.value()};
}

template <typename T>
void slider(std::string_view parameter_id, T &value, T min, T max,
	    T reset_value, bool is_disabled = false,
	    std::string_view label_dimension = "",
	    std::string_view base_label = "");

template <typename T>
void slider(std::string_view group_id, std::string_view parameter_id, T &value,
            T min, T max, T reset_value, bool is_disabled = false,
            std::string_view label_dimension = "",
	    std::string_view base_label = "") {
  auto nested_id = fmt::format("{}.{}", group_id, parameter_id);
  slider(std::string_view(nested_id), value, min, max, reset_value, is_disabled,
	 label_dimension, base_label);
}

template <typename T>
bool vector_table(
    std::string_view group_id, std::string_view parameter_label, T &vec,
    typename T::vector_type min, typename T::vector_type max,
    std::array<typename T::vector_type, types::VectorSize<T>::value>
	reset_values,
    std::array<bool, types::VectorSize<T>::value> disabled = {},
    std::array<std::string, types::VectorSize<T>::value> labels = {});

template <typename T>
bool vector_table(
    std::string_view group_id, std::string_view parameter_label, T &vec,
    typename T::vector_type min, typename T::vector_type max, T reset_value,
    std::array<bool, types::VectorSize<T>::value> disabled = {},
    std::array<std::string, types::VectorSize<T>::value> labels = {}) {

  constexpr auto vector_size = types::VectorSize<T>::value;
  std::array<typename T::vector_type, vector_size> reset_values;

  for (std::size_t i = 0; i < vector_size; ++i) {
    reset_values[i] = reset_value[i];
  }

  return vector_table(group_id, parameter_label, vec, min, max, reset_values,
                      disabled, labels);
}

template <typename T>
bool vector_table(std::string_view group_id, std::string_view parameter_label,
		  T &vec, typename T::vector_type min,
                  typename T::vector_type max,
                  typename T::vector_type reset_value,
                  bool disabled_value = false) {

  constexpr auto vector_size = types::VectorSize<T>::value;

  std::array<typename T::vector_type, vector_size> reset_values;
  reset_values.fill(reset_value);

  std::array<bool, vector_size> disabled_array;
  disabled_array.fill(disabled_value);

  return vector_table(group_id, parameter_label, vec, min, max, reset_values,
                      disabled_array);
}

bool begin_tree_node(std::string_view name, bool &open);

void tween_config(std::string_view label,
                  pc::tween::TweenConfiguration &config);

} // namespace pc::gui
