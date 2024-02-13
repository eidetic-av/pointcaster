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
#include <type_traits>
#include <variant>
#include <map>

namespace pc {

using pc::types::Float2;
using pc::types::Float3;
using pc::types::Float4;
using pc::types::Int2;
using pc::types::Int3;
using pc::types::MinMax;
using pc::types::Short2;
using pc::types::Short3;

template <typename T>
concept is_float_vector_t =
    std::disjunction<std::is_same<T, Float2>, std::is_same<T, Float3>,
  std::is_same<T, Float4>, std::is_same<T, MinMax<float>>>::value;

template <typename T>
concept is_int_vector_t =
    std::disjunction<std::is_same<T, Int2>, std::is_same<T, Int3>,
		     std::is_same<T, MinMax<int>>>::value;

template <typename T>
concept is_short_vector_t =
    std::disjunction<std::is_same<T, Short2>, std::is_same<T, Short3>,
		     std::is_same<T, MinMax<short>>>::value;

struct Parameter;

using FloatReference = std::reference_wrapper<float>;
using Float2Reference = std::reference_wrapper<Float2>;
using Float3Reference = std::reference_wrapper<Float3>;
using IntReference = std::reference_wrapper<int>;
using Int2Reference = std::reference_wrapper<Int2>;
using Int3Reference = std::reference_wrapper<Int3>;
using ShortReference = std::reference_wrapper<short>;
using Short2Reference = std::reference_wrapper<Short2>;
using Short3Reference = std::reference_wrapper<Short3>;

using MinMaxFloatReference = std::reference_wrapper<MinMax<float>>;
using MinMaxIntReference = std::reference_wrapper<MinMax<int>>;
using MinMaxShortReference = std::reference_wrapper<MinMax<short>>;

using BoolReference = std::reference_wrapper<bool>;
using StringReference = std::reference_wrapper<std::string>;

using ParameterReference =
    std::variant<FloatReference, Float2Reference, Float3Reference, IntReference,
		 Int2Reference, Int3Reference, ShortReference, Short2Reference, Short3Reference, MinMaxFloatReference,
		 MinMaxIntReference, MinMaxShortReference, BoolReference,
		 StringReference>;

using ParameterValue =
    std::variant<float, Float2, Float3, Float4, int, Int2, Int3, short, Short2,
		 Short3, MinMax<float>, MinMax<int>, MinMax<short>, bool,
		 std::string>;

using ParameterUpdateCallback =
    std::function<void(const Parameter &, Parameter &)>;

struct Parameter {
  ParameterReference value; 
  ParameterValue min;
  ParameterValue max;
  ParameterValue default_value;

  std::string parent_struct_name = "";

  std::vector<ParameterUpdateCallback> update_callbacks;
  std::vector<ParameterUpdateCallback> minmax_update_callbacks;
  std::vector<std::function<void()>> erase_callbacks;

  Parameter(float &_value, float _min = -10, float _max = 10,
            float _default_value = 0)
      : value(FloatReference(_value)), min(_min), max(_max),
	default_value(_default_value) {}

  Parameter(Float2 &_value, float _min = -10, float _max = 10,
            float _default_value = 0)
      : value(Float2Reference(_value)), min(_min), max(_max),
        default_value(_default_value) {}

  Parameter(Float3 &_value, float _min = -10, float _max = 10,
            float _default_value = 0)
      : value(Float3Reference(_value)), min(_min), max(_max),
        default_value(_default_value) {}

  Parameter(int &_value, int _min = -10, int _max = 10, int _default_value = 0)
      : value(IntReference(_value)), min(_min), max(_max),
        default_value(_default_value) {}

  Parameter(Int2 &_value, int _min = -10, int _max = 10, int _default_value = 0)
      : value(Int2Reference(_value)), min(_min), max(_max),
	default_value(_default_value) {}

  Parameter(short &_value, short _min = -10, short _max = 10,
            short _default_value = 0)
      : value(ShortReference(_value)), min(_min), max(_max),
        default_value(_default_value) {}

  Parameter(Short2 &_value, short _min = -10, short _max = 10,
            short _default_value = 0)
      : value(Short2Reference(_value)), min(_min), max(_max),
        default_value(_default_value) {}

  Parameter(Short3 &_value, short _min = -10, short _max = 10,
            short _default_value = 0)
      : value(Short3Reference(_value)), min(_min), max(_max),
        default_value(_default_value) {}

  Parameter(MinMax<float> &_value, float _min = -10, float _max = 10,
            float _default_value = 0)
      : value(MinMaxFloatReference(_value)), min(_min), max(_max),
        default_value(_default_value) {}

  Parameter(MinMax<int> &_value, int _min = -10, int _max = 10,
            int _default_value = 0)
      : value(MinMaxIntReference(_value)), min(_min), max(_max),
        default_value(_default_value) {}

  Parameter(MinMax<short> &_value, short _min = -10, short _max = 10,
	    short _default_value = 0)
      : value(MinMaxShortReference(_value)), min(_min), max(_max),
	default_value(_default_value) {}

  Parameter(bool &_value)
      : value(BoolReference(_value)), min(false), max(true),
	default_value(false) {}

  Parameter(std::string &_value)
      : value(StringReference(_value)), min(""), max(""), default_value("") {}

  float current() const {
    if (std::holds_alternative<FloatReference>(value)) {
      return std::get<FloatReference>(value).get();
    } else if (std::holds_alternative<IntReference>(value)) {
      return std::get<IntReference>(value).get();
    } else {
      pc::logger->warn(
          "Attempted to retrieve 'current' value of complex parameter type.");
      return 0.0f;
    }
  }
};

enum class ParameterState { Unbound, Bound, Learning };

inline pc::string_map<Parameter> parameter_bindings;
inline pc::string_map<ParameterState> parameter_states;

struct ParameterMap;

using NestedParameterMap = std::pair<std::string, ParameterMap>;
using ParameterMapEntry = std::variant<std::string, NestedParameterMap>;

struct ParameterMap : public std::vector<ParameterMapEntry> {};

inline std::map<std::string, ParameterMap> struct_parameters;

void declare_parameter(std::string_view parameter_id,
		       const Parameter &parameter_binding);

template <typename T>
void declare_parameters(std::string_view parameter_id, T &basic_value,
			T default_value, std::optional<MinMax<float>> min_max,
			std::string_view parent_struct_name) {

  // numeric values
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int> ||
		std::is_same_v<T, short>) {
    Parameter p(basic_value);
    p.parent_struct_name = parent_struct_name;
    p.default_value = default_value;
    if (min_max.has_value()) {
      p.min = static_cast<T>(min_max->min);
      p.max = static_cast<T>(min_max->max);
    } else {
      p.min = T(-10);
      p.max = T(10);
    }
    declare_parameter(parameter_id, std::move(p));
  }
  // bools and strings
  else if constexpr (std::is_same_v<T, bool> ||
                     std::is_same_v<T, std::string>) {
    Parameter p(basic_value);
    p.parent_struct_name = parent_struct_name;
    p.default_value = default_value;
    declare_parameter(parameter_id, std::move(p));
  }
  // ignore enums for now
  else if constexpr (std::is_enum_v<T>)
    return;
}

template <typename T>
  requires pc::types::VectorType<T>
void declare_parameters(std::string_view parameter_id, T &vector_value,
			T default_value, std::optional<MinMax<float>> min_max,
			std::string_view parent_struct_name) {

  constexpr auto vector_size = types::VectorSize<T>::value;
  constexpr std::array<const char *, 4> element = {"x", "y", "z", "w"};

  // we declare the vector itself as a parameter, as it is
  // considered one parameter when rendering the gui
  Parameter p(vector_value);
  p.parent_struct_name = parent_struct_name;
  p.default_value = default_value;
  if (min_max.has_value()) {
    p.min = static_cast<typename T::vector_type>(min_max->min);
    p.max = static_cast<typename T::vector_type>(min_max->max);
  } else {
    p.min = (typename T::vector_type)(-10);
    p.max = (typename T::vector_type)(10);
  }
  declare_parameter(parameter_id, std::move(p));

  // we also declare the vector elements as individual parameters,
  // which can be targetted individually by inputs like MIDI, etc.
  for (int i = 0; i < vector_size; i++) {
    auto element_id = fmt::format("{}.{}", parameter_id, element[i]);
    // pc::logger->debug("declare vector: {}", element_id);
    declare_parameters(element_id, vector_value[i], default_value[i], min_max,
		       parent_struct_name);
  }
}

template <typename T>
  requires pc::reflect::IsSerializable<T>
void declare_parameters(std::string_view parameter_id, T &complex_value,
			T default_value = {},
			std::optional<MinMax<float>> min_max = {},
			std::string_view parent_struct_name = "") {

  // retrieve type info at compile-time
  constexpr auto type_info = serde::type_info<T>;
  constexpr auto struct_name = T::Name;

  const auto member_names = type_info.member_names().members();
  const auto member_defaults = T::Defaults;
  const auto member_min_max_values = T::MinMaxValues;

  // create an integer sequence to iterate through the type's members at
  // compile-time... this is required because the index to the member is used
  // inside a template argument
  constexpr auto index_sequence = std::make_index_sequence<T::MemberCount>{};

  // Function to apply on each type member.
  auto handle_member = [parameter_id, struct_name, &complex_value, &type_info,
			&member_defaults, &member_min_max_values](
			   std::string_view member_name, auto index) {
    using MemberType =
        pc::reflect::type_at_t<typename T::MemberTypes, decltype(index)::value>;

    // retrieve the actual member reference at runtime
    auto &member_ref =
        type_info.template member<decltype(index)::value>(complex_value);

    // and any more metadata (default value, min+max values)
    auto member_default = std::get<index>(member_defaults);
    auto member_min_max = member_min_max_values.at(index);

    // recursively call declare_parameters for each member, delegating to the
    // appropriate overload based on type
    auto member_parameter_id = fmt::format("{}.{}", parameter_id, member_name);
    declare_parameters(member_parameter_id, member_ref, member_default, member_min_max,
		       struct_name);
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
		    const Parameter &parameter);

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
