#pragma once

#include "../point_cloud.h"
#include "driver.h"

namespace bob::sensors {

template <class driver_t> class Device {

public:
  Device() { _driver.open(); }

  ~Device() {
    if (_driver.isOpen())
      _driver.close();
  }

  virtual void spin() = 0;

protected:
  driver_t _driver;
};

} // namespace bob::sensors
