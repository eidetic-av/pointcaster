#pragma once

#include <algorithm>
#include <concepts>
#include <limits>

namespace math {

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
  if (!clamp) return mapped_result;
  return std::min(std::max(mapped_result, new_min), new_max);
};

} // namespace math
