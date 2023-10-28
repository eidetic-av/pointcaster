#include "parameters.h"
#include "logger.h"

namespace pc {

void declare_parameter(std::string_view parameter_id,
		       const ParameterBinding &parameter_binding) {
  parameter_bindings.emplace(parameter_id, parameter_binding);
}

void bind_parameter(std::string_view parameter_id,
                    const ParameterBinding &parameter_binding) {
  declare_parameter(parameter_id, parameter_binding);
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

  auto &binding = parameter_bindings.at(parameter_id);
  auto old_binding = binding;

  if (std::holds_alternative<FloatReference>(binding.value)) {
    float &value = std::get<FloatReference>(binding.value).get();
    value = math::remap(input_min, input_max, binding.min, binding.max,
                        new_value, true);
  } else {
    int &value = std::get<IntReference>(binding.value).get();
    auto float_value = math::remap(input_min, input_max, binding.min,
                                   binding.max, new_value, true);
    value = static_cast<int>(std::round(float_value));
  }

  for (const auto &cb : binding.update_callbacks) {
    cb(old_binding, binding);
  }
}

} // namespace pc
