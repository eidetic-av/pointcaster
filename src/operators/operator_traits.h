#pragma once

#include <concepts>

namespace pc::operators {

template <typename T, typename Scene, typename Group>
concept requires_init = requires(T config, Scene &scene, Group &group) {
  { T::init(config, scene, group) } -> std::same_as<void>;
};

template <typename, typename = std::void_t<>>
struct has_draw : std::false_type {};

template <typename T>
struct has_draw<T, std::void_t<decltype(std::declval<T>().draw)>>
    : std::true_type {};

template <typename T> inline constexpr bool has_draw_v = has_draw<T>::value;

} // namespace pc::operators