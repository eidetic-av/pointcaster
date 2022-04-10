#pragma once

#include "../device.h"
#include "k4a_driver.h"

namespace bob::sensors {

class K4ADevice : public Device {
public:
  K4ADevice();
  ~K4ADevice();

  bob::PointCloud getPointCloud() override { return _driver->getPointCloud(); };
  std::string getBroadcastId() override;
};
} // namespace bob::sensors
