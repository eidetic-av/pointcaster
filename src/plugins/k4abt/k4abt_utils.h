#pragma once

#include "../../structs.h"
#include <array>
#include <cmath>
#include <k4a/k4a.hpp>
#include <k4abt.hpp>

namespace pc::k4a_utils {

using pc::types::Float3;
using pc::types::Float4;

inline std::array<Float3, 32>
calculateAverageJointPositions(const std::vector<k4abt_skeleton_t> &skeletons) {
  std::array<Float3, 32> averages;
  for (int joint_id = 0; joint_id < 32; joint_id++) {
    Float3 sum{0, 0, 0};
    for (const auto &skeleton : skeletons) {
      const auto joint = skeleton.joints[joint_id];
      const auto position = joint.position.xyz;
      sum.x += position.x;
      sum.y += position.y;
      sum.z += position.z;
    }
    auto skeleton_count = skeletons.size();
    averages[joint_id] = {sum.x / skeleton_count, sum.y / skeleton_count,
                          sum.z / skeleton_count};
  }
  return averages;
}

inline std::array<Float4, 32> calculateAverageJointOrientations(
    const std::vector<k4abt_skeleton_t> &skeletons) {
  std::array<Float4, 32> averages;
  for (int joint_id = 0; joint_id < 32; joint_id++) {
    Float4 sum{0, 0, 0, 0};
    for (const auto &skeleton : skeletons) {
      const auto joint = skeleton.joints[joint_id];
      const auto orientation = joint.orientation.wxyz;
      sum.w += orientation.w;
      sum.x += orientation.x;
      sum.y += orientation.y;
      sum.z += orientation.z;
    }
    float k = 1.0f / std::sqrt(sum.w * sum.w + sum.x * sum.x + sum.y * sum.y +
                               sum.z * sum.z);
    averages[joint_id] = {sum.w * k, sum.x * k, sum.y * k, sum.z * k};
  }
  return averages;
}

} // namespace pc::k4a_utils
