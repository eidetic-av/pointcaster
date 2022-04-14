#pragma once

#include "../../point_cloud.h"
#include "../driver.h"
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <spdlog/spdlog.h>
#include <thread>

namespace bob::sensors {

class K4ADriver : public Driver {
public:
  K4ADriver(int device_index_ = 0);
  ~K4ADriver();

  bool open() override;
  bool close() override;

  bool isOpen() override { return _open; };

  PointCloud getPointCloud() override;

  std::string getId() override {
    if (_serial_number.empty())
      _serial_number = _device.get_serialnum();
    return _serial_number;
  }

private:
  k4a::device _device;
  k4a_device_configuration_t _config;
  k4a::calibration _calibration;
  k4a::transformation _transformation;
  std::string _serial_number;

  std::mutex _buffer_mutex;
  std::atomic<bool> _buffers_updated = false;
  std::vector<int16_t> _positions_buffer;
  std::vector<uint8_t> _colors_buffer;
  size_t _point_count;

  PointCloud _point_cloud{std::vector<position>{{0, 0, 0, 0}},
			  std::vector<float>{1}};

  bool _open = false;
};
} // namespace bob::sensors