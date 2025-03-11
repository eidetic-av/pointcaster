#pragma once

#include <concepts>

namespace pc::operators {

template <typename T, typename Scene, typename Group>
concept RequiresInit = requires(T config, Scene &scene, Group &group) {
  { T::init(config, scene, group) } -> std::same_as<void>;
};

template <typename T>
concept ToggleableDrawing = requires(const T &t) {
  { t.draw } -> std::convertible_to<bool>;
};

} // namespace pc::operators