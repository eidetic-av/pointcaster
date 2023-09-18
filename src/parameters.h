#pragma once

#include "logger.h"
#include "math.h"
#include "serialization.h"
#include "string_map.h"
#include "structs.h"
#include <any>
#include <cmath>
#include <concepts>
#include <fmt/format.h>
#include <functional>
#include <iostream>
#include <optional>
#include <serdepp/adaptor/reflection.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <variant>

namespace pc {

using pc::types::float2;
using pc::types::float3;
using pc::types::float4;
using pc::types::int2;
using pc::types::int3;

struct ParameterBinding;

using FloatReference = std::reference_wrapper<float>;
using IntReference = std::reference_wrapper<int>;

using ParameterReference = std::variant<FloatReference, IntReference>;

using ParameterUpdateCallback =
    std::function<void(const ParameterBinding &, ParameterBinding &)>;

struct ParameterBinding {
  ParameterReference value; 
  float min;
  float max;

  std::vector<ParameterUpdateCallback> update_callbacks;
  std::vector<ParameterUpdateCallback> minmax_update_callbacks;
  std::vector<std::function<void()>> erase_callbacks;

  ParameterBinding(float &val, float min_val = -10, float max_val = 10)
      : value(FloatReference(val)), min(min_val), max(max_val) {}

  ParameterBinding(int &val, float min_val = -10, float max_val = 10)
      : value(IntReference(val)), min(min_val), max(max_val) {}

  float current() const {
    if (std::holds_alternative<FloatReference>(value)) {
      return std::get<FloatReference>(value).get();
    } else {
      return std::get<IntReference>(value).get();
    }
  }
};

enum class ParameterState { Unbound, Bound, Learning };

inline pc::string_map<ParameterBinding> parameter_bindings;
inline pc::string_map<ParameterState> parameter_states;

struct ReflectedParameterInfo {
  std::string name;
  std::any ref;
  ReflectedParameterInfo(std::string_view _name, ParameterReference _ref)
      : name(_name), ref(_ref){};
};

void declare_parameter(std::string_view parameter_id,
		       const ParameterBinding &parameter_binding);

template <typename T>
void declare_parameters(std::string_view parameter_id,
                        T &basic_value) {
  // ignore strings -- they can't be externally updated
  if constexpr (std::is_same_v<T, std::string>) return;

  // TODO we should probably handle bools and enums, but ignore them for now
  if constexpr (std::is_enum_v<T> || std::is_same_v<T, bool>) return;

  // Handle floats and integers
  if constexpr (std::is_same_v<T, float>) {
    pc::logger->debug("declare float: {}", parameter_id);
    declare_parameter(parameter_id, ParameterBinding(basic_value));
  } else if constexpr (std::is_same_v<T, int>) {
    pc::logger->debug("declare int: {}", parameter_id);
    declare_parameter(parameter_id, ParameterBinding(basic_value));
  }
}

template <typename T>
  requires pc::types::IsVectorType<T>
void declare_parameters(std::string_view parameter_id, T &vector_value) {

  constexpr auto vector_size = types::VectorSize<T>::value;
  constexpr std::array<const char *, 4> element = {"x", "y", "z", "w"};

  // for a vector type, we just loop through each element and sent the reference
  // to the basic_value overload as a float or int
  for (int i = 0; i < vector_size; i++) {
    auto element_id = fmt::format("{}.{}", parameter_id, element[i]);
    pc::logger->debug("declare vector: {}", element_id);
    declare_parameters(element_id, vector_value[i]);
  }
}

template <typename T>
  requires pc::reflect::IsSerializable<T>
void declare_parameters(std::string_view parameter_id, T &complex_value) {

  // retrieve type info at compile-time
  constexpr auto type_info = serde::type_info<T>;

  // member names are determined at runtime
  const auto member_names = type_info.member_names().members();

  // create an integer sequence to iterate through the type's members at
  // compile-time... this isrequired because the index to the member is used
  // inside a template argument
  constexpr auto index_sequence = std::make_index_sequence<T::MemberCount>{};

  // Function to apply on each type member.
  auto handle_member = [parameter_id, &complex_value,
                        &type_info](std::string_view member_name, auto index) {
    using MemberType =
        pc::reflect::type_at_t<typename T::MemberTypes, decltype(index)::value>;

    // retrieve the actual member reference at runtime
    auto &member_ref =
        type_info.template member<decltype(index)::value>(complex_value);

    // recursively call declare_parameters for each member, delegating to the
    // appropriate overload based on type
    auto member_parameter_id = fmt::format("{}.{}", parameter_id, member_name);
    pc::logger->debug("declare: {}", member_parameter_id);
    declare_parameters(member_parameter_id, member_ref);
  };

  // using an immediately invoked lambda to provide member names and
  // the associated index as a compile-time constant to the handler function
  [&handle_member,
   &member_names]<std::size_t... Is>(std::index_sequence<Is...>) {
    ((handle_member(member_names[Is],
                    std::integral_constant<std::size_t, Is>{})),
     ...);
  }(index_sequence);
}

void bind_parameter(std::string_view parameter_id,
		    const ParameterBinding &parameter_binding);

void unbind_parameter(std::string_view parameter_id);

void set_parameter_value(std::string_view parameter_id,
                                float new_value, float input_min,
				float input_max);

inline void set_parameter_value(std::string_view parameter_id, int new_value,
				int input_min, int input_max) {
  set_parameter_value(parameter_id, static_cast<float>(new_value),
		      static_cast<float>(input_min),
		      static_cast<float>(input_max));
}

/* Returns a copy of the current parameter value */
inline float get_parameter_value(std::string_view parameter_id) {
  return parameter_bindings.at(parameter_id).current();
}

inline void set_parameter_minmax(std::string_view parameter_id, float min,
                              float max) {
  auto &binding = parameter_bindings.at(parameter_id);
  binding.min = min;
  binding.max = max;
}

inline void add_parameter_update_callback(std::string_view parameter_id,
                                       ParameterUpdateCallback callback) {
  auto &binding = parameter_bindings.at(parameter_id);
  binding.update_callbacks.push_back(std::move(callback));
}

inline void add_parameter_minmax_update_callback(std::string_view parameter_id,
                                              ParameterUpdateCallback callback) {
  auto &binding = parameter_bindings.at(parameter_id);
  binding.minmax_update_callbacks.push_back(std::move(callback));
}

inline void add_parameter_erase_callback(std::string_view parameter_id,
					 std::function<void()> callback) {
  auto &binding = parameter_bindings.at(parameter_id);
  binding.erase_callbacks.push_back(std::move(callback));
}

inline void clear_parameter_callbacks(std::string_view parameter_id) {
  auto &binding = parameter_bindings.at(parameter_id);
  binding.update_callbacks.clear();
  binding.minmax_update_callbacks.clear();
  binding.erase_callbacks.clear();
}

} // namespace pc
