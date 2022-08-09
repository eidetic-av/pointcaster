#pragma once

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

  bob::types::PointCloud getPointCloud(const DeviceConfiguration& config) override;

  std::string getId() override {
    if (_serial_number.empty())
      _serial_number = _device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    return _serial_number;
  }

  rs2::device getDevice() { return _device; }

  rs2::depth_sensor getDepthSensor() {
      return _device.first<rs2::depth_sensor>();
  }

  rs2::color_sensor getColorSensor() {
      return _device.first<rs2::color_sensor>();
  }

private:
  rs2::context _context;
  rs2::config _config;
  rs2::pipeline _pipe;
  rs2::device _device;
  std::string _serial_number;

  std::mutex _buffer_mutex;
  std::atomic<bool> _buffers_updated;
  std::vector<float> _positions_buffer;
  size_t _point_count;

  std::vector<rs2::texture_coordinate> _uvs_buffer;
  std::vector<uint8_t> _colors_buffer;

  int _texture_width;
  int _texture_height;
  size_t _texture_pixel_size;
  size_t _texture_stride;

  bob::types::PointCloud _point_cloud;

  bool _open = false;
};
} // namespace bob::sensors
