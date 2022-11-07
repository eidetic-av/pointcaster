#pragma once

#include "../device.h"
#include "k4a_driver.h"

namespace bob::sensors {

class K4ADevice : public Device {
public:
  K4ADevice();
  ~K4ADevice();

  std::string getBroadcastId() override;

  void drawDeviceSpecificControls() override;
  void updateDeviceControl(int *target, int value,
			   std::function<void(int)> set_func);

private:
  int _exposure = 33330;
  int _brightness = 128;
  int _contrast = 5;
  int _saturation = 32;
  int _gain = 128;

};
} // namespace bob::sensors
