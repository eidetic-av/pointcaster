#pragma once

#include "../structs.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <vector>
#include <string>

namespace bob::sensors {

class Driver {
public:
  int device_index;

  virtual ~Driver() = default;

  virtual bool isOpen() const = 0;

  virtual bob::types::PointCloud pointCloud(const bob::types::DeviceConfiguration& config) = 0;
  virtual std::string id() const = 0;

  bool primary_aligner = false;

  virtual void startAlignment() = 0;
  virtual bool isAligning() = 0;
  virtual bool isAligned() = 0;
};

} // namespace bob::sensors
