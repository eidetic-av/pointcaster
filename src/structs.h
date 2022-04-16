#pragma once

struct minMax {
  float min = -10;
  float max = 10;

  float* arr() { return &min; }

  bool contains(float value) {
    if (value < min) return false;
    if (value > max) return false;
    return true;
  }
};

struct float3 {
  float x, y, z = 0;
};

struct position {
  float x, y, z, __pad = 0;
};

struct DeviceConfiguration {
  bool flip_x, flip_y, flip_z;
  minMax crop_x, crop_y, crop_z;
  position offset;
  float scale;
};
