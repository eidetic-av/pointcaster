#pragma once

#include "../structs.h"
#include "device_config.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <string>
#include <vector>

namespace pc::devices {

class Driver {
public:
  uint device_index;
  bool primary_aligner = false;

  virtual ~Driver() = default;

  virtual void start_sensors() = 0;
  virtual void stop_sensors() = 0;
  virtual void reload() = 0;
  virtual void reattach() = 0;

  virtual bool is_open() const = 0;
  virtual bool is_running() const = 0;

  virtual pc::types::PointCloud point_cloud(const DeviceConfiguration &config) = 0;
  virtual std::string id() const = 0;

  virtual void set_paused(bool pause) = 0;

  virtual void start_alignment() = 0;
  virtual bool is_aligning() = 0;
  virtual bool is_aligned() = 0;
};

} // namespace pc::devices
