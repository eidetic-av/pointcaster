#pragma once

#include <k4a/k4a.h>
#include "../driver.h"
#include "../../point_cloud.h"

namespace bob::sensors {

  class K4ADriver : Driver {
  public:
    int device_index;

    K4ADriver(int _device_index = 0);

    bool Open() override;
    bool Close() override;

    bool IsOpen() override {
      return _open;
    };

    PointCloud<Vector3, float> getPointCloud();

  private:
    k4a_device_t _device;
    k4a_device_configuration_t _config;
    k4a_calibration_t _calibration;
    k4a_transformation_t _transformation;

    bool _open = false;

  };
} //namespace bob::sensors
