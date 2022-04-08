#include "k4a_device.h"

namespace bob::sensors {

  using namespace Magnum;

  void K4ADevice::spin() {

  }

  std::vector<Vector3> K4ADevice::getPointCloud() {
    return _driver.getPointCloud();
  }
  
} // namespace bob::sensors
