#pragma once

#include <k4a/k4a.h>
#include "../driver.h"
#include "../../point_cloud.h"
#include <thread>

namespace bob::sensors {

  class K4ADriver : Driver {
  public:
    int device_index;

    K4ADriver(int device_index_ = 0);

    bool open() override;
    bool close() override;

    bool isOpen() override {
      return _open;
    };

    PointCloud getPointCloud();

  private:
    k4a_device_t _device;
    k4a_device_configuration_t _config;
    k4a_calibration_t _calibration;
    k4a_transformation_t _transformation;

    std::mutex _buffer_mutex;
    std::vector<int16_t> _positions_buffer;
    std::vector<uint8_t> _colors_buffer;
    size_t _point_count;
    bool _buffers_updated = false;

    PointCloud _point_cloud {
      std::vector<Vector3>{ },
      std::vector<float>{ 1 }
    };

    bool _open = false;

  };
} //namespace bob::sensors
