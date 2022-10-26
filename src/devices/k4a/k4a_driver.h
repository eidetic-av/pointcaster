#pragma once

#include "../driver.h"
#include "../device.h"
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <k4abt.hpp>
#include <spdlog/spdlog.h>
#include <thread>
#include <array>
#include <Eigen/Geometry>

namespace bob::sensors {

class K4ADriver : public Driver {
public:
  static constexpr uint2 color_resolution { 1280, 720 };
  static constexpr uint2 depth_resolution { 512, 512 };

  K4ADriver(int device_index_ = 0);
  ~K4ADriver();

  std::string id() const override;
  bool isOpen() const override;

  void startAlignment() override;
  bool isAligning() override;
  bool isAligned() override;

  bob::types::PointCloud pointCloud(const DeviceConfiguration& config) override;
  
private:
  static constexpr uint incoming_point_count =
      depth_resolution.x * depth_resolution.y;
  static constexpr uint color_buffer_size =
      incoming_point_count * sizeof(color);
  static constexpr uint positions_buffer_size =
      incoming_point_count * sizeof(short3);

  std::string serial_number;

  bool stop_requested = false;
  std::thread _capture_loop;
  k4a::device device;
  k4a_device_configuration_t _config;
  k4a::calibration _calibration;
  k4a::transformation _transformation;
  std::string _serial_number;

  std::mutex _buffer_mutex;
  std::atomic<bool> _buffers_updated;

  std::array<short3, positions_buffer_size> positions_buffer;
  std::array<color, color_buffer_size> colors_buffer;

  k4abt::tracker tracker;
  static constexpr uint total_alignment_frames = 10;
  uint alignment_frame_count = total_alignment_frames;
  std::vector<k4abt_skeleton_t> alignment_skeleton_frames;
  bool aligned = false;
  short3 alignment_center {0, 0, 0};
  short3 aligned_position_offset;
  Eigen::Quaternion<float> aligned_orientation_offset;

  void runAligner(const k4a::capture &frame);

  bob::types::PointCloud _point_cloud;

  bool _open = false;
  };
} // namespace bob::sensors
