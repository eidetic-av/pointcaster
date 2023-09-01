#pragma once

#include "../structs.h"
#include "device_config.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <string>
#include <vector>

namespace pc::sensors {

class Driver {
public:
  uint device_index;
  bool primary_aligner = false;

  virtual ~Driver() = default;

  virtual const bool is_open() const = 0;

  virtual pc::types::PointCloud point_cloud(const DeviceConfiguration &config) = 0;
  virtual std::string id() const = 0;

  virtual void set_paused(bool pause) = 0;

  virtual void start_alignment() = 0;
  virtual bool is_aligning() = 0;
  virtual bool is_aligned() = 0;
};

} // namespace pc::sensors
