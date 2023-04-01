#pragma once

#include "../driver.h"
#include "../device.h"
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <k4abt.hpp>
#include <thread>
#include <array>
#include <exception>
#include <Eigen/Geometry>

namespace bob::sensors {

class K4ADriver : public Driver {
public:
  static constexpr bob::types::uint2 color_resolution { 1280, 720 };
  static constexpr bob::types::uint2 depth_resolution { 512, 512 };

  K4ADriver(int device_index_ = 0);
  ~K4ADriver();

  std::string id() const override;
  bool isOpen() const override;

  void setPaused(bool paused) override;

  void startAlignment() override;
  bool isAligning() override;
  bool isAligned() override;

  bob::types::PointCloud pointCloud(const bob::types::DeviceConfiguration& config) override;

  void setExposure(const int new_exposure);
  const int getExposure() const;
  void setBrightness(const int new_brightness);
  const int getBrightness() const;
  void setContrast(const int new_contrast);
  const int getContrast() const;
  void setSaturation(const int new_saturation);
  const int getSaturation() const;
  void setGain(const int new_gain);
  const int getGain() const;

private:
  static constexpr uint incoming_point_count =
      depth_resolution.x * depth_resolution.y;
  static constexpr uint color_buffer_size =
      incoming_point_count * sizeof(bob::types::color);
  static constexpr uint positions_buffer_size =
      incoming_point_count * sizeof(bob::types::short3);

  std::string serial_number;

  std::atomic<bool> pause_sensor = false;

  bool stop_requested = false;
  std::thread _capture_loop;
  k4a::device device;
  k4a_device_configuration_t _config;
  k4a::calibration _calibration;
  k4a::transformation _transformation;
  std::string _serial_number;

  std::mutex _buffer_mutex;
  std::atomic<bool> _buffers_updated;

  std::array<bob::types::short3, positions_buffer_size> positions_buffer;
  std::array<bob::types::color, color_buffer_size> colors_buffer;

  k4abt::tracker tracker;
  static constexpr uint total_alignment_frames = 10;
  uint alignment_frame_count = total_alignment_frames;
  std::vector<k4abt_skeleton_t> alignment_skeleton_frames;
  bool aligned = false;
  bob::types::position alignment_center {0, 0, 0};
  bob::types::position aligned_position_offset;
  Eigen::Quaternion<float> aligned_orientation_offset;

  void runAligner(const k4a::capture &frame);

  bob::types::PointCloud _point_cloud;

  bool _open = false;
  };
} // namespace bob::sensors
