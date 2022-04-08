#pragma once

#include "../device.h"
#include "k4a_driver.h"

namespace bob::sensors {

  class K4ADevice : Device<K4ADriver> {
  public:
    void spin() override;
    std::vector<Magnum::Vector3> getPointCloud() override;
  };
  
}
