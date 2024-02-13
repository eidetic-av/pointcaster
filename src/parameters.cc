#include "parameters.h"
#include "logger.h"
#include "string_utils.h"
#include <stack>

namespace pc {

void declare_parameter(std::string_view parameter_id,
                       const Parameter &parameter) {
  parameter_bindings.emplace(parameter_id, parameter);

  // Ignored suffixes check
  static constexpr std::array<std::string, 4> ignored_suffixes = {".x", ".y",
                                                                  ".z", ".w"};
  if (pc::strings::ends_with_any(parameter_id, ignored_suffixes.begin(),
                                 ignored_suffixes.end())) {
    return;
  }

  std::size_t start = 0;
  std::size_t end = parameter_id.find('.');
  std::string struct_name{parameter_id.substr(start, end - start)};

  if (!struct_parameters.contains(struct_name)) {
    struct_parameters.emplace(struct_name, ParameterMap{});
  }

  std::stack<std::reference_wrapper<ParameterMap>> map_stack;
  map_stack.push(struct_parameters[struct_name]);
  start = end + 1;

  while ((end = parameter_id.find('.', start)) != std::string_view::npos) {
    std::string level_name =
        std::string(parameter_id.substr(start, end - start));
    bool level_exists = false;

    ParameterMap &current_map = map_stack.top();
    for (auto &entry : current_map) {
      if (std::holds_alternative<NestedParameterMap>(entry)) {
        auto &nested_parameter_map = std::get<NestedParameterMap>(entry);
        auto &nested_level_name = nested_parameter_map.first;

        if (nested_level_name == level_name) {
          map_stack.push(nested_parameter_map.second);
          level_exists = true;
          break;
        }
      }
    }

    if (!level_exists) {
      ParameterMap new_map;
      current_map.emplace_back(NestedParameterMap{level_name, new_map});
      map_stack.push(new_map);
    }

    start = end + 1;
  }

  // add the parameter to the deepest map
  ParameterMap &deepest_map = map_stack.top();
  deepest_map.emplace_back(std::string(parameter_id));
}

void bind_parameter(std::string_view parameter_id,
                    const Parameter &parameter) {
  declare_parameter(parameter_id, parameter);
  parameter_states.emplace(parameter_id, ParameterState::Bound);
}

void unbind_parameter(std::string_view parameter_id) {
  auto it = parameter_bindings.find(parameter_id);
  if (it != parameter_bindings.end()) {
    auto &binding = it->second;
    for (auto &erase_cb : binding.erase_callbacks) {
      erase_cb();
    }
  }
  // parameter_bindings.erase(parameter_id);
  parameter_states[parameter_id] = ParameterState::Unbound;
}

void set_parameter_value(std::string_view parameter_id, float new_value,
			 float input_min, float input_max) {

  auto &parameter = parameter_bindings.at(parameter_id);
  auto old_binding = parameter;

  if (std::holds_alternative<FloatReference>(parameter.value)) {
    float &value = std::get<FloatReference>(parameter.value).get();
    value = math::remap(input_min, input_max, std::get<float>(parameter.min),
			std::get<float>(parameter.max), new_value, true);
  } else if (std::holds_alternative<IntReference>(parameter.value)) {
    int &value = std::get<IntReference>(parameter.value).get();
    auto min = static_cast<float>(std::get<int>(parameter.min));
    auto max = static_cast<float>(std::get<int>(parameter.max));
    auto float_value =
	math::remap(input_min, input_max, min, max, new_value, true);
    value = static_cast<int>(std::round(float_value));
  } else if (std::holds_alternative<ShortReference>(parameter.value)) {
    short &value = std::get<ShortReference>(parameter.value).get();
    auto min = static_cast<float>(std::get<short>(parameter.min));
    auto max = static_cast<float>(std::get<short>(parameter.max));
    auto float_value =
	math::remap(input_min, input_max, min, max, new_value, true);
    value = static_cast<short>(std::round(float_value));
  }

  for (const auto &cb : parameter.update_callbacks) {
    cb(old_binding, parameter);
  }
}

} // namespace pc
