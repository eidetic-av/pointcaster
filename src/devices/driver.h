#pragma once

#include "../structs.h"
#include "../operators/session_operator_host.h"
#include "device_config.gen.h"
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <string>
#include <vector>

// shouldn't really include k4a types, but this whole driver needs to disappear
// anyway as its /only/ used for k4a types at the moment, it should be
// consolidated into the k4a device behaviour, with alignment occuring in operators
#include "k4a/k4a_config.gen.h"

namespace pc::devices {

class Driver {
public:
  unsigned int device_index;
  bool primary_aligner = false;
  bool lost_device = false;

  virtual ~Driver() = default;

  virtual void start_sensors() = 0;
  virtual void stop_sensors() = 0;
  virtual void reload() = 0;

  virtual bool is_open() const = 0;
  virtual bool is_running() const = 0;

  virtual pc::types::PointCloud
  point_cloud(AzureKinectConfiguration &config,
              pc::operators::OperatorList operators = {}) = 0;
  virtual std::string id() const = 0;

  virtual void set_paused(bool pause) = 0;

  virtual void start_alignment() = 0;
  virtual bool is_aligning() = 0;
  virtual bool is_aligned() = 0;
};

} // namespace pc::devices
