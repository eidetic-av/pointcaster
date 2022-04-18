#pragma once

#include "../point_cloud.h"
#include "../structs.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <vector>

namespace bob::sensors {

class Driver {
public:
  int device_index;

  virtual ~Driver() = default;

  virtual bool open() = 0;
  virtual bool close() = 0;
  virtual bool isOpen() = 0;

  virtual PointCloud getPointCloud(const DeviceConfiguration& config) = 0;
  virtual std::string getId() = 0;
};

} // namespace bob::sensors
