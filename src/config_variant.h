#pragma once

#include <concepts>
#include <type_traits>
#include <utility>
#include <variant>

#include <camera/camera_config.h>
#include <session/session_config.h>

namespace pc {

// ---------- generic utilities ----------

template <typename T, typename Variant> struct is_in_variant;

template <typename T, typename... Ts>
struct is_in_variant<T, std::variant<Ts...>>
    : std::bool_constant<(std::same_as<T, Ts> || ...)> {};

template <typename T, typename Variant>
inline constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;

template <typename Callback, typename Variant>
constexpr void for_each_variant_type(Callback cb) {
  constexpr std::size_t count = std::variant_size_v<Variant>;
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (cb.template operator()<std::variant_alternative_t<I, Variant>>(), ...);
  }(std::make_index_sequence<count>{});
}

// ---------- core configuration variant ----------

using CoreConfigurationVariant =
    std::variant<CameraConfiguration, SessionConfiguration
                 /* WorkspaceConfiguration, ... */
                 >;

template <typename T>
concept ValidCoreConfig = is_in_variant_v<T, CoreConfigurationVariant>;

template <typename Callback>
constexpr void for_each_core_config_type(Callback cb) {
  for_each_variant_type<Callback, CoreConfigurationVariant>(cb);
}

} // namespace pc
