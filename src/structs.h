#pragma once
#include <pointclouds.h>

namespace bob::types {

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

struct float3 {
  float x, y, z = 0;
};

struct float4 {
  float w, x, y, z = 0;
};

struct int3 {
  int x, y, z = 0;
};

struct DeviceConfiguration {
  bool flip_x, flip_y, flip_z;
  minMax<short> crop_x{-10000, 10000};
  minMax<short> crop_y{-10000, 10000};
  minMax<short> crop_z{-10000, 10000};
  short3 offset;
  float3 rotation_deg{0, 0, 0};
  float scale;
};

} // namespace bob::types
