#pragma once

#include "../device.h"
#include "k4a_driver.h"

namespace bob::sensors {

class K4ADevice : public Device {
public:
  K4ADevice();
  ~K4ADevice();

  std::string getBroadcastId() override;
};
} // namespace bob::sensors
