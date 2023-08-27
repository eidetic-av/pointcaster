#pragma once

#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Vector3.h>
#include <algorithm>
#include <limits>
#include <numbers>

namespace pc::math {

#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif

template <typename T>
using number = std::enable_if_t<std::is_arithmetic<T>::value, T>;

/**
 * @brief Remap a value from one range to another.
 *
 * @tparam T A type that must be a numeric type
 *
 * @param old_min The lower bound of the value's current range
 * @param old_max The upper bound of the value's current range
 * @param new_min The lower bound of the value's target range
 * @param new_max The upper bound of the value's target range
 * @param value The value to be remapped
 *
 * @return The value remapped to the target range
 *
 * @note If old_min is equal to old_max, new_min is returned.
 */
template <typename T, typename U>
constexpr auto remap(T old_min, T old_max, U new_min, U new_max, U value,
		     bool clamp = false)
    -> std::enable_if_t<
	std::is_arithmetic<T>::value && std::is_floating_point<U>::value, U> {
  static_assert(std::is_arithmetic<T>::value, "T must be numeric");
  static_assert(std::is_floating_point<U>::value, "U must be floating point");

  if (old_min == old_max)
    return new_min;

  const auto denominator =
      std::max((old_max - old_min), std::numeric_limits<U>::epsilon());
  const auto mapped_result =
      (value - old_min) / denominator * (new_max - new_min) + new_min;

  if (!clamp)
    return mapped_result;

  return std::min(std::max(mapped_result, new_min), new_max);
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
inline constexpr auto degToRad(T degrees) {
  return degrees * (M_PI / static_cast<T>(180));
}

inline constexpr Magnum::Math::Rad<float> degToRad(Magnum::Math::Deg<float> degrees) {
  return Magnum::Math::Rad<float>{degrees};
}

inline Magnum::Math::Vector3<Magnum::Math::Rad<float>>
degToRad(Magnum::Math::Vector3<Magnum::Math::Deg<float>> degrees) {
  return {Magnum::Math::Rad<float>{degrees.x()},
	  Magnum::Math::Rad<float>{degrees.y()},
	  Magnum::Math::Rad<float>{degrees.z()}};
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
inline constexpr auto radToDeg(T radians) {
  return radians * (static_cast<T>(180) / M_PI);
}

inline constexpr Magnum::Math::Deg<float> radToDeg(Magnum::Math::Rad<float> radians) {
  return Magnum::Math::Deg<float>{radians};
}

inline Magnum::Math::Vector3<Magnum::Math::Deg<float>>
radToDeg(Magnum::Math::Vector3<Magnum::Math::Rad<float>> radians) {
  return {Magnum::Math::Deg<float>{radians.x()},
	  Magnum::Math::Deg<float>{radians.y()},
	  Magnum::Math::Deg<float>{radians.z()}};
}

} // namespace pc::math
