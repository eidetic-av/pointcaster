#pragma once

#include <pointcaster/core_types.h>
#include <rfl/DefaultVal.hpp>
#include <rfl/Skip.hpp>
#include <rfl/internal/to_ptr_named_tuple.hpp>

#include <functional>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace pc {

// all scalar types representable inside configs...
// vector types & other configs are aggregates of these
using ConfigValue = std::variant<bool, int, float, double, std::string>;

class ConfigRegistry {
public:
  struct Field {
    std::function<ConfigValue()> get;
    std::function<void(ConfigValue)> set;
  };

  using ChangeCallback = std::function<void(std::string_view path)>;

  void register_field(std::string path, Field field);

  // Remove all registered paths that begin with prefix
  void unregister(std::string_view prefix);

  void clear();

  // returns false if path not found
  [[maybe_unused]] bool set(std::string_view path, ConfigValue value);

  std::optional<ConfigValue> get(std::string_view path) const;

  void on_change(std::string_view prefix, ChangeCallback cb);

  void remove_subscriptions(std::string_view prefix);

private:
  mutable std::shared_mutex _mutex;
  std::unordered_map<std::string, Field> _fields;
  std::vector<std::pair<std::string, ChangeCallback>> _subs;

  void notify(std::string_view path);
};

// ---- rfl wrapper type traits ----

template <class T> struct is_rfl_default_val : std::false_type {};
template <class T>
struct is_rfl_default_val<rfl::DefaultVal<T>> : std::true_type {};
template <class T>
inline constexpr bool is_rfl_default_val_v = is_rfl_default_val<T>::value;

template <class T> struct is_rfl_skip : std::false_type {};
template <class T> struct is_rfl_skip<rfl::Skip<T>> : std::true_type {};
template <class T> inline constexpr bool is_rfl_skip_v = is_rfl_skip<T>::value;

// Types that map to a ConfigValue variant arm directly
template <class T>
concept ConfigValueCompatible =
    std::is_same_v<T, bool> || std::is_same_v<T, int> ||
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::string> || std::is_enum_v<T>;

// Types rfl can traverse recursively (plain aggregates; excludes arrays,
// std::string, std::vector, etc. which are not aggregates)
template <class T>
concept RflTraversable = std::is_aggregate_v<T> && !std::is_array_v<T>;

// ---- Value conversion ----

template <class T> ConfigValue to_config_value(const T &v) {
  if constexpr (std::is_same_v<T, bool>)
    return v;
  else if constexpr (std::is_same_v<T, int>)
    return v;
  else if constexpr (std::is_same_v<T, float>)
    return v;
  else if constexpr (std::is_same_v<T, double>)
    return v;
  else if constexpr (std::is_same_v<T, std::string>)
    return v;
  else if constexpr (std::is_enum_v<T>)
    return static_cast<int>(v);
  else
    static_assert(sizeof(T) == 0, "to_config_value: unsupported type");
}

template <class T> T from_config_value(const ConfigValue &v) {
  if constexpr (std::is_enum_v<T>) {
    // enums come in as int, but accept any numeric
    return std::visit(
        [](auto &&val) -> T {
          using V = std::decay_t<decltype(val)>;
          if constexpr (std::is_arithmetic_v<V>)
            return static_cast<T>(static_cast<int>(val));
          return T{};
        },
        v);
  } else if constexpr (std::is_same_v<T, bool>) {
    return std::visit(
        [](auto &&val) -> bool {
          using V = std::decay_t<decltype(val)>;
          if constexpr (std::is_arithmetic_v<V>) return static_cast<bool>(val);
          if constexpr (std::is_same_v<V, std::string>) return !val.empty();
          return false;
        },
        v);
  } else if constexpr (std::is_arithmetic_v<T>) {
    // int/float/double all coerce freely to each other
    return std::visit(
        [](auto &&val) -> T {
          using V = std::decay_t<decltype(val)>;
          if constexpr (std::is_arithmetic_v<V>) return static_cast<T>(val);
          if constexpr (std::is_same_v<V, std::string>) {
            try {
              if constexpr (std::is_floating_point_v<T>)
                return static_cast<T>(std::stod(val));
              else
                return static_cast<T>(std::stol(val));
            } catch (...) {
              return T{};
            }
          }
          return T{};
        },
        v);
  } else if constexpr (std::is_same_v<T, std::string>) {
    return std::visit(
        [](auto &&val) -> std::string {
          using V = std::decay_t<decltype(val)>;
          if constexpr (std::is_same_v<V, std::string>) return val;
          if constexpr (std::is_same_v<V, bool>) return val ? "true" : "false";
          if constexpr (std::is_arithmetic_v<V>) return std::to_string(val);
          return {};
        },
        v);
  } else {
    return std::get<T>(v);
  }
}

// ---- register_config ----

// Forward declaration so nested calls compile
template <class T>
void register_config(ConfigRegistry &reg, std::string_view prefix, T &cfg);

template <class T>
void register_config(ConfigRegistry &reg, std::string_view prefix, T &cfg) {
  auto ptr_nt = rfl::internal::to_ptr_named_tuple(cfg);
  ptr_nt.apply([&](auto field) {
    using FieldType = std::decay_t<decltype(field)>;
    using ValPtrType = typename FieldType::Type; // e.g. rfl::DefaultVal<int>*
    using ValType =
        std::remove_pointer_t<ValPtrType>; // e.g. rfl::DefaultVal<int>

    if constexpr (is_rfl_skip_v<ValType>) return;

    const std::string path =
        std::string(prefix) + "/" + std::string(FieldType::name());
    ValType *ptr = field.value_;

    if constexpr (is_rfl_default_val_v<ValType>) {
      using Inner = typename ValType::Type;
      if constexpr (RflTraversable<Inner>) {
        register_config(reg, path, ptr->value());
      } else if constexpr (ConfigValueCompatible<Inner>) {
        reg.register_field(
            path,
            {[ptr]() -> ConfigValue { return to_config_value(ptr->value()); },
             [ptr](ConfigValue v) { ptr->set(from_config_value<Inner>(v)); }});
      }
      // else: vector, variant, etc. inside DefaultVal — skip
    } else if constexpr (RflTraversable<ValType>) {
      register_config(reg, path, *ptr);
    } else if constexpr (ConfigValueCompatible<ValType>) {
      reg.register_field(
          path,
          {[ptr]() -> ConfigValue { return to_config_value(*ptr); },
           [ptr](ConfigValue v) { *ptr = from_config_value<ValType>(v); }});
    }
    // else: std::vector, std::unordered_map, etc. — skip silently
  });
}

} // namespace pc