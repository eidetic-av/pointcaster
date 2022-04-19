#pragma once

#include "../device.h"
#include "rs2_driver.h"

namespace bob::sensors {

class Rs2Device : public Device {
public:
  Rs2Device();
  ~Rs2Device();

  std::string getBroadcastId() override;
};
} // namespace bob::sensors
