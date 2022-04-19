#pragma once

#include "../../point_cloud.h"
#include "../driver.h"
#include "../device.h"
#include <spdlog/spdlog.h>
#include <thread>

#include <librealsense2/rs.hpp>

namespace bob::sensors {

class Rs2Driver : public Driver {
public:
  Rs2Driver(int device_index_ = 0);
  ~Rs2Driver();

  bool open() override;
  bool close() override;

  bool isOpen() override { return _open; };

  PointCloud getPointCloud(const DeviceConfiguration& config) override;

  std::string getId() override {
    if (_serial_number.empty())
      _serial_number = _device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    return _serial_number;
  }

private:
  rs2::context _context;
  rs2::config _config;
  rs2::pipeline _pipe;
  rs2::device _device;
  std::string _serial_number;

  std::mutex _buffer_mutex;
  std::atomic<bool> _buffers_updated = false;
  std::vector<float> _positions_buffer;
  std::vector<uint8_t> _colors_buffer;
  size_t _point_count;

  PointCloud _point_cloud{std::vector<position>{{0, 0, 0, 0}},
			  std::vector<float>{1}};

  bool _open = false;
};
} // namespace bob::sensors
