#include "parameters.h"
#include "logger.h"
#include "string_utils.h"
#include "utils/lru_cache.h"
#include <algorithm>
#include <stack>
#include <type_traits>

namespace pc::parameters {

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
  parameter_states[parameter_id] = ParameterState::Unbound;
}

void publish_parameter(std::string_view parameter_id) {
  pc::logger->debug("running publish parameter: {}", parameter_id);
  parameter_states[parameter_id] = ParameterState::Publish;
}

template <typename T> cache::lru_cache<std::string, T> &get_cache() {
  static constexpr auto parameter_cache_size = 50;
  static cache::lru_cache<std::string, T> cache(parameter_cache_size);
  return cache;
}

void publish() {
  for (const auto &kvp : parameter_states) {
    const auto &[parameter_id, state] = kvp;

    if (state == ParameterState::Publish) {
      std::visit(
          [parameter_id](auto &&parameter_ref) {

	    const auto &p = parameter_ref.get();
            using T = std::decay_t<decltype(p)>;

	    auto& param_cache = get_cache<T>();

	    bool value_updated = true;
	    if (param_cache.exists(parameter_id)) {
	      const auto &cached_value = param_cache.get(parameter_id);
	      value_updated = p != cached_value;
	    }

	    if (value_updated) {

	      if constexpr (is_publishable_container_v<T>) {
		publisher::publish_all(parameter_id, p);
	      } else if constexpr (pc::types::VectorType<T>) {
		constexpr auto vector_size = types::VectorSize<T>::value;
		std::array<typename T::vector_type, vector_size> array;
		std::copy_n(p.data(), vector_size, array.begin());
		publisher::publish_all(parameter_id, array);
	      }
	      // TODO MQTT doesn't implement single parameter publishing
	      // else if constexpr (std::is_same_v<float, T>) {
	      // 	publisher::publish_all(parameter_id, p);
	      // } else if constexpr (std::is_same_v<int, T>) {
	      // 	publisher::publish_all(parameter_id, p);
	      // }
	      param_cache.put(parameter_id, p);
            }
          },
          parameter_bindings.at(parameter_id).value);
    }
  }
}

} // namespace pc::parameters
