#include "parameters.h"
#include "logger.h"
#include "session.gen.h"
#include "string_utils.h"
#include "utils/lru_cache.h"
#include <algorithm>
#include <functional>
#include <stack>
#include <type_traits>
#include <unordered_map>

namespace pc::parameters {

void declare_parameter(std::string_view parameter_id,
                       const Parameter &parameter) {
  pc::logger->debug(" -- param: {}", parameter_id.data());
  auto [_, new_binding] =
      parameter_bindings.insert_or_assign(std::string{parameter_id}, parameter);

  if (!new_binding) {
    // just updating that parameter reference
    return;
  }

  // otherwise if its a new binding, we need to recurse into the parameter and
  // declare nested parameter structs

  // Ignored suffixes check
  static const std::array<std::string, 4> ignored_suffixes = {"/x", "/y", "/z",
                                                              "/w"};
  if (pc::strings::ends_with_any(parameter_id, ignored_suffixes.begin(),
                                 ignored_suffixes.end())) {
    return;
  }

  std::size_t start = 0;
  std::size_t end = parameter_id.find('/');
  std::string struct_name{parameter_id.substr(start, end - start)};

  if (!struct_parameters.contains(struct_name)) {
    struct_parameters.emplace(struct_name, ParameterMap{});
  }

  std::stack<std::reference_wrapper<ParameterMap>> map_stack;
  map_stack.push(struct_parameters[struct_name]);
  start = end + 1;

  while ((end = parameter_id.find('/', start)) != std::string_view::npos) {
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

void bind_parameter(std::string_view parameter_id, const Parameter &parameter) {
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

void unbind_parameters(std::string_view parameter_id) {
  std::erase_if(parameter_bindings.inner_map(), [parameter_id](auto &kvp) {
    return pc::strings::starts_with(std::get<0>(kvp), parameter_id);
  });
  std::erase_if(parameter_states.inner_map(), [parameter_id](auto &kvp) {
    return pc::strings::starts_with(std::get<0>(kvp), parameter_id);
  });
  std::erase_if(struct_parameters, [parameter_id](auto &kvp) {
    return pc::strings::starts_with(std::get<0>(kvp), parameter_id);
  });
}

void publish_parameter(std::string_view parameter_id) {
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
    if (!parameter_bindings.contains(parameter_id)) continue;
    if (state == ParameterState::Publish) {
      auto &binding = parameter_bindings.at(parameter_id);
      const auto &session_label =
          session_label_from_id[binding.host_session_id].empty()
              ? binding.host_session_id
              : session_label_from_id[binding.host_session_id];
      std::visit(
          [parameter_id, session_label](auto &&parameter_ref) {
            const auto &p = parameter_ref.get();
            using T = std::decay_t<decltype(p)>;

            auto &param_cache = get_cache<T>();

            bool value_updated = true;
            if (param_cache.exists(parameter_id)) {
              const auto &cached_value = param_cache.get(parameter_id);
              value_updated = p != cached_value;
            }

            if (value_updated) {
              if constexpr (is_publishable_container_v<T>) {
                publisher::publish_all(parameter_id, p, {session_label});
              } else if constexpr (pc::types::VectorType<T>) {
                constexpr auto vector_size = types::VectorSize<T>::value;
                std::array<typename T::vector_type, vector_size> array;
                std::copy_n(p.data(), vector_size, array.begin());
                publisher::publish_all(parameter_id, array, {session_label});
              } else if constexpr (pc::types::ScalarType<T>) {
                publisher::publish_all(parameter_id, p, {session_label});
              }
              param_cache.put(parameter_id, p);
            }
          },
          binding.value);
    }
  }
}

} // namespace pc::parameters
