#pragma once

#include <vector>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>

namespace bob::sensors {

class Driver {
public:
  virtual bool Open() = 0;
  virtual bool Close() = 0;
  virtual bool IsOpen() = 0;

  virtual std::vector<Magnum::Vector3> getPointCloud() = 0;
};

} // namespace bob::sensors
