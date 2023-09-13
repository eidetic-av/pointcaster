#pragma once

#include "../fonts/IconsFontAwesome6.h"
#include "../math.h"
#include "../structs.h"
#include "../tween/tween_config.h"
#include "range_slider.h"
#include <array>
#include <atomic>
#include <functional>
#include <imgui.h>
#include <iostream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace pc::gui {

enum ParameterType { Float, Int };

constexpr float ParamUndeclared = -99.99f;

struct GuiParameter {
  void *value;
  ParameterType param_type;
  float range_min;
  float range_max;
};

/* Must be called at the start of each imgui frame */
void begin_gui_helpers();

/* Used for generating unique ids for imgui parameters */
extern unsigned int _parameter_index;

template <typename T>
void draw_slider(std::string_view label_text, T *value, T min, T max,
                 T default_value = 0);

struct ParameterBinding;

using ParameterUpdateCallback =
    std::function<void(const ParameterBinding &, ParameterBinding &)>;

enum class ParameterState { Unbound, Bound, Recording };

inline std::unordered_map<std::string, ParameterState> parameter_states;

struct ParameterBinding {
  float value = ParamUndeclared;
  float min = ParamUndeclared;
  float max = ParamUndeclared;

  std::vector<ParameterUpdateCallback> update_callbacks;
  std::vector<ParameterUpdateCallback> minmax_update_callbacks;
};

inline std::unordered_map<std::string, ParameterBinding> parameter_bindings;

inline std::atomic_bool recording_parameter;
inline std::atomic<ParameterState> recording_result;
inline std::string recording_parameter_id;
inline std::pair<float, float> recording_parameter_minmax;
inline std::mutex recording_parameter_mutex;

inline void store_recording_parameter_info(const std::string &id, float min,
                                        float max) {
  std::lock_guard<std::mutex> lock(recording_parameter_mutex);
  recording_parameter_id = id;
  recording_parameter_minmax = {min, max};
}

inline std::pair<std::string, std::pair<float, float>>
load_recording_parameter_info() {
  std::lock_guard<std::mutex> lock(recording_parameter_mutex);
  return {recording_parameter_id, recording_parameter_minmax};
}

inline void add_parameter_update_callback(const std::string &parameter_id,
                                       ParameterUpdateCallback callback) {
  auto &binding = parameter_bindings[parameter_id];
  binding.update_callbacks.push_back(std::move(callback));
}

inline void add_parameter_minmax_update_callback(const std::string &parameter_id,
                                              ParameterUpdateCallback callback) {
  auto &binding = parameter_bindings[parameter_id];
  binding.minmax_update_callbacks.push_back(std::move(callback));
}

inline void set_parameter_value(const std::string &parameter_id, float value,
                             float input_min, float input_max) {
  auto &binding = parameter_bindings[parameter_id];
  auto old_binding = binding;

  binding.value = pc::math::remap(input_min, input_max, binding.min,
                                  binding.max, value, true);

  for (const auto &cb : binding.update_callbacks) {
    cb(old_binding, binding);
  }
}

inline void set_parameter_value(const std::string &parameter_id, int value,
                             int input_min, int input_max) {
  set_parameter_value(parameter_id, static_cast<float>(value),
                   static_cast<float>(input_min),
                   static_cast<float>(input_max));
}

inline float get_parameter_value(const std::string &parameter_id) {
  return parameter_bindings[parameter_id].value;
}

inline void set_parameter_minmax(const std::string &parameter_id, float min,
                              float max) {
  auto &binding = parameter_bindings[parameter_id];
  binding.min = min;
  binding.max = max;
}

template <typename T>
void parameter(const std::string &parameter_id, std::size_t i, T &value, T min, T max,
            T reset_value, bool is_disabled, const std::string &label_dimension,
            const std::string &base_label);

template <typename T, std::size_t N>
bool vector_table(
    const std::string group_id, const std::string label, std::array<T, N> &vec,
    T min, T max, std::array<T, N> reset_values,
    std::array<bool, N> disabled = std::array<bool, N>{false},
    std::array<std::string, N> labels = std::array<std::string, N>{""});

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

void tween_config(std::string_view label,
                  pc::tween::TweenConfiguration &config);

} // namespace pc::gui
