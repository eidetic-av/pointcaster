#pragma once

#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Vector3.h>
#include <algorithm>
#include <concepts>
#include <limits>
#include <numbers>

namespace pc::math {

template <typename T>
concept number = std::is_arithmetic_v<T>;

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
constexpr auto remap(number auto old_min, number auto old_max,
                     number auto new_min, number auto new_max,
                     std::floating_point auto value, bool clamp = false) {
  if (old_min == old_max)
    return new_min;
  const auto denominator = std::max(
      (old_max - old_min), std::numeric_limits<decltype(new_min)>::epsilon());
  const auto mapped_result =
      (value - old_min) / denominator * (new_max - new_min) + new_min;
  if (!clamp)
    return mapped_result;
  return std::min(std::max(mapped_result, new_min), new_max);
};

constexpr auto degToRad(std::floating_point auto degrees) {
  return degrees * (std::numbers::pi_v<decltype(degrees)> /
                    static_cast<decltype(degrees)>(180));
}

constexpr Magnum::Math::Rad<float> degToRad(Magnum::Math::Deg<float> degrees) {
  return Magnum::Math::Rad<float>{degrees};
}

Magnum::Math::Vector3<Magnum::Math::Rad<float>>
degToRad(Magnum::Math::Vector3<Magnum::Math::Deg<float>> degrees) {
  return {Magnum::Math::Rad<float>{degrees.x()},
	  Magnum::Math::Rad<float>{degrees.y()},
	  Magnum::Math::Rad<float>{degrees.z()}};
}

constexpr auto radToDeg(std::floating_point auto radians) {
  return radians * (static_cast<decltype(radians)>(180) /
                    std::numbers::pi_v<decltype(radians)>);
}

constexpr Magnum::Math::Deg<float> radToDeg(Magnum::Math::Rad<float> radians) {
  return Magnum::Math::Deg<float>{radians};
}

Magnum::Math::Vector3<Magnum::Math::Deg<float>>
radToDeg(Magnum::Math::Vector3<Magnum::Math::Rad<float>> radians) {
  return {Magnum::Math::Deg<float>{radians.x()},
	  Magnum::Math::Deg<float>{radians.y()},
	  Magnum::Math::Deg<float>{radians.z()}};
}

} // namespace pc::math
