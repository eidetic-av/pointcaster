#pragma once

#include "../device.h"
#include "k4a_driver.h"

namespace bob::sensors {

  class K4ADevice : Device<K4ADriver> {
  public:
    void spin() override;
    bob::PointCloud<Vector3, float> getPointCloud() {
      return _driver.getPointCloud();
    };
  };
  
}
