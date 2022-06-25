#pragma once

struct minMax {
  float min = -10;
  float max = 10;

  float* arr() { return &min; }

  bool contains(float value) const {
    if (value < min) return false;
    if (value > max) return false;
    return true;
  }
};

struct float3 {
  float x, y, z = 0;
};

struct short3 {
  short x, y, z = 0;
};

struct DeviceConfiguration {
  bool flip_x, flip_y, flip_z;
  minMax crop_x, crop_y, crop_z;
  float3 offset;
  float scale;
};
