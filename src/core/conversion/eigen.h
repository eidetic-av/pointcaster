#pragma once

#include <pointcaster/core_types.h>

#include <Eigen/Core>

namespace pc {

inline float4x4 to_float4x4(const Eigen::Matrix4f &matrix) {
  float4x4 output{};

  for (Eigen::Index row = 0; row < 4; ++row) {
    for (Eigen::Index column = 0; column < 4; ++column) {
      output.values[static_cast<std::size_t>(row * 4 + column)] =
          matrix(row, column);
    }
  }

  return output;
}

} // namespace pc