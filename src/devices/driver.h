#pragma once

#include "../point_cloud.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <vector>

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

namespace bob::sensors {

class Driver {
public:
  int device_index;
  minMax crop_x;
  minMax crop_y;
  minMax crop_z;

  virtual bool open() = 0;
  virtual bool close() = 0;
  virtual bool isOpen() = 0;

  virtual PointCloud getPointCloud() = 0;
  virtual std::string getId() = 0;
};

} // namespace bob::sensors
