#pragma once

#include "../device.h"
#include "k4a_driver.h"

namespace pc::sensors {

class K4ADevice : public Device {
public:
  K4ADevice();
  ~K4ADevice();

  std::string get_broadcast_id() override;

  void draw_device_controls() override;
  void update_device_control(int *target, int value,
                             std::function<void(int)> set_func);

private:
  int _exposure = 33330;
  int _brightness = 128;
  int _contrast = 5;
  int _saturation = 32;
  int _gain = 128;
};
} // namespace pc::sensors
