#pragma once

#include "../point_cloud.h"
#include "driver.h"

namespace bob::sensors {

template <class driver_t> class Device {

public:
  Device() { _driver.Open(); }

  ~Device() {
    if (_driver.IsOpen())
      _driver.Close();
  }

  virtual void spin() = 0;

protected:
  driver_t _driver;
};

} // namespace bob::sensors
