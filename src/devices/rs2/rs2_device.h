#pragma once

#include "../device.h"
#include "rs2_driver.h"

namespace pc::sensors {

class Rs2Device : public Device {
public:
  Rs2Device();
  ~Rs2Device();

  std::string getBroadcastId() override;

protected:
  void drawDeviceSpecificControls() override;
};
} // namespace pc::sensors
