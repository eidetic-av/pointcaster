#pragma once

#include <vector>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include "../point_cloud.h"

namespace bob::sensors {

class Driver {
public:
  virtual bool open() = 0;
  virtual bool close() = 0;
  virtual bool isOpen() = 0;

  // virtual PointCloud getPointCloud() = 0;
};

} // namespace bob::sensors
