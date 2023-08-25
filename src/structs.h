#pragma once

#include <pointclouds.h>
#include <nlohmann/json.hpp>

namespace pc::types {

template <typename T> struct minMax {
  T min;
  T max;

  T *arr() { return &min; }

  bool contains(T value) const {
    if (value < min)
      return false;
    if (value > max)
      return false;
    return true;
  }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(minMax<float>, min, max);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(minMax<double>, min, max);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(minMax<int>, min, max);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(minMax<unsigned int>, min, max);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(minMax<short>, min, max);

struct short3 {
  short x, y, z = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(short3, x, y, z);

struct float3 {
  float x, y, z = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(float3, x, y, z);

struct float4 {
  float w, x, y, z = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(float4, w, x, y, z);

struct int3 {
  int x, y, z = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(int3, x, y, z);

// just alias these for now before refactoring bob-pointclouds library
using PointCloud = bob::types::PointCloud;
using position = bob::types::position;
using color = bob::types::color;
using uint2 = bob::types::uint2;

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(color, r, g, b, a);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(uint2, x, y);

} // namespace pc::types
