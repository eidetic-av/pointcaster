#pragma once

#include <pointclouds.h>

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

struct short3 {
  short x, y, z = 0;
};

struct float3 {
  float x, y, z = 0;
};

struct float4 {
  float w, x, y, z = 0;
};

struct int3 {
  int x, y, z = 0;
};

// just alias these for now before refactoring bob-pointclouds library
using PointCloud = bob::types::PointCloud;
using position = bob::types::position;
using color = bob::types::color;
using uint2 = bob::types::uint2;

} // namespace pc::types
