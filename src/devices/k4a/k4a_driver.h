#pragma once

#include "../device.h"
#include "../driver.h"
#include <Eigen/Geometry>
#include <array>
#include <exception>
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <k4abt.hpp>
#include <thread>

namespace pc::sensors {

using pc::types::color;
using pc::types::PointCloud;
using pc::types::position;
using pc::types::short3;
using pc::types::uint2;

class K4ADriver : public Driver {

public:
  static constexpr uint2 color_resolution{1280, 720};
  static constexpr uint2 depth_resolution{512, 512};

  static uint device_count;

  K4ADriver();
  ~K4ADriver();

  std::string id() const override;
  bool is_open() const override;

  void set_paused(bool paused) override;

  PointCloud point_cloud(const DeviceConfiguration &config) override;

  void start_alignment() override;
  bool is_aligning() override;
  bool is_aligned() override;

  void set_exposure(const int new_exposure);
  int get_exposure() const;
  void set_brightness(const int new_brightness);
  int get_brightness() const;
  void set_contrast(const int new_contrast);
  int get_contrast() const;
  void set_saturation(const int new_saturation);
  int get_saturation() const;
  void set_gain(const int new_gain);
  int get_gain() const;

  std::array<float, 3> accelerometer_sample();
  void apply_auto_tilt(const bool apply);

private:
  static constexpr uint incoming_point_count =
      depth_resolution.x * depth_resolution.y;
  static constexpr uint color_buffer_size =
      incoming_point_count * sizeof(color);
  static constexpr uint positions_buffer_size =
      incoming_point_count * sizeof(short3);

  std::string _serial_number;

  std::atomic<bool> _pause_sensor = false;
  bool _stop_requested = false;
  std::thread _capture_loop;
  k4a::device _device;
  k4a_device_configuration_t _config;
  k4a::calibration _calibration;
  k4a::transformation _transformation;

  std::mutex _buffer_mutex;
  std::atomic<bool> _buffers_updated;

  std::array<short3, positions_buffer_size> _positions_buffer;
  std::array<color, color_buffer_size> _colors_buffer;

  k4abt::tracker _tracker;
  static constexpr uint _total_alignment_frames = 10;
  uint _alignment_frame_count = _total_alignment_frames;
  std::vector<k4abt_skeleton_t> _alignment_skeleton_frames;
  bool _aligned = false;
  position _alignment_center{0, 0, 0};
  position _aligned_position_offset;
  Eigen::Quaternion<float> _aligned_orientation_offset;

  void run_aligner(const k4a::capture &frame);

  Eigen::Matrix3f _auto_tilt = Eigen::Matrix3f::Identity();

  PointCloud _point_cloud;

  bool _open = false;
};
} // namespace pc::sensors
