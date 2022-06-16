#pragma once

#include "../../point_cloud.h"
#include "../driver.h"
#include "../device.h"
#include <spdlog/spdlog.h>
#include <thread>

namespace bob::sensors {

class K4W2Driver : public Driver {
public:
  K4W2Driver(int device_index_ = 0);
  ~K4W2Driver();

  bool open() override;
  bool close() override;

  bool isOpen() override { return _open; };

  PointCloud getPointCloud(const DeviceConfiguration& config) override;

  std::string getId() override {
    if (_serial_number.empty())
      _serial_number = "unset";
    return _serial_number;
  }

private:
  //static libfreenect2::Freenect2 _freenect2;
  //libfreenect2::Freenect2Device* _device;
  //libfreenect2::PacketPipeline* _pipeline;
  std::string _serial_number;

  std::mutex _buffer_mutex;
  std::atomic<bool> _buffers_updated;
  std::vector<int16_t> _positions_buffer;
  std::vector<uint8_t> _colors_buffer;
  size_t _point_count;

  PointCloud _point_cloud{std::vector<position>{{0, 0, 0, 0}},
			  std::vector<float>{1}};

  bool _open = false;
};
} // namespace bob::sensors
