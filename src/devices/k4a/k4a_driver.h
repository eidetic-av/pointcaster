#pragma once

#include "../device.h"
#include "../driver.h"
#include "../device_config.h"

#include <Eigen/Geometry>
#include <array>
#include <exception>
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <k4abt.hpp>
#include <thread>
#include <atomic>

namespace pc::devices {

using pc::transformers::TransformerList;
using pc::types::color;
using pc::types::Float3;
using pc::types::Float4;
using pc::types::PointCloud;
using pc::types::position;
using pc::types::Short3;
using pc::types::Uint2;

using K4ASkeleton =
    std::array<std::pair<pc::types::position, Float4>, K4ABT_JOINT_COUNT>;

// Forward declaration hides CUDA types, allowing K4ADriver to have CUDA
// members. This prevents issues when this header is included in TUs not
// compiled with nvcc.
struct K4ADriverImplDeviceMemory;

class K4ADriver : public Driver {

public:
  static constexpr Uint2 color_resolution{1280, 720};
  static constexpr Uint2 depth_resolution{512, 512};
  static constexpr std::size_t incoming_point_count =
      depth_resolution.x * depth_resolution.y;

  static inline std::atomic<uint> active_count = 0;

  K4ADriver(const DeviceConfiguration& config);
  ~K4ADriver();

  std::string id() const override;
  bool is_open() const override;
  bool is_running() const override;

  void start_sensors() override;
  void stop_sensors() override;
  void reload() override;
  void reattach() override;

  void set_paused(bool paused) override;

  PointCloud point_cloud(const DeviceConfiguration &config,
			 TransformerList transformers = {}) override;

  void start_alignment() override;
  bool is_aligning() override;
  bool is_aligned() override;

  void enable_body_tracking(const bool enabled);
  bool tracking_bodies() { return _body_tracking_enabled; };
  std::vector<K4ASkeleton> skeletons() { return _skeletons; };

  void set_depth_mode(const k4a_depth_mode_t mode);

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

  void set_auto_tilt(bool enabled) { _auto_tilt_enabled = enabled; };
  void clear_auto_tilt() { _auto_tilt = Eigen::Matrix3f::Identity(); }

private:

  static constexpr uint color_buffer_size =
      incoming_point_count * sizeof(color);
  static constexpr uint positions_buffer_size =
      incoming_point_count * sizeof(Short3);
  static constexpr uint positions_in_size =
      sizeof(Short3) * incoming_point_count;
  static constexpr uint positions_out_size =
      sizeof(position) * incoming_point_count;
  static constexpr uint colors_size = sizeof(color) * incoming_point_count;

  K4ADriverImplDeviceMemory *_device_memory;
  std::atomic_bool _device_memory_ready{false};
  void init_device_memory();
  void free_device_memory();

  std::atomic_bool _open{false};
  std::atomic_bool _running{false};

  std::string _serial_number;
  DeviceConfiguration _last_config;

  std::unique_ptr<k4a::device> _device;
  std::unique_ptr<k4abt::tracker> _tracker;

  std::atomic_bool _pause_sensor{false};
  bool _stop_requested = false;
  std::thread _capture_loop;

  k4a_device_configuration_t _k4a_config;
  k4a::calibration _calibration;
  k4a::transformation _transformation;

  std::mutex _buffer_mutex;
  std::atomic<bool> _buffers_updated;

  std::array<Short3, incoming_point_count> _positions_buffer;
  std::array<color, incoming_point_count> _colors_buffer;

  bool _body_tracking_enabled;
  std::thread _tracker_loop;
  std::vector<K4ASkeleton> _skeletons;

  static constexpr uint _total_alignment_frames = 10;
  uint _alignment_frame_count = _total_alignment_frames;
  std::vector<k4abt_skeleton_t> _alignment_skeleton_frames;
  bool _aligned = false;
  position _alignment_center{0, 0, 0};
  position _aligned_position_offset;
  Eigen::Quaternion<float> _aligned_orientation_offset;

  std::thread _imu_loop;
  bool _auto_tilt_enabled;
  Eigen::Matrix3f _auto_tilt = Eigen::Matrix3f::Identity();

  void capture_frames();
  void track_bodies();
  void process_imu();
  void run_aligner(const k4a::capture &frame);

  void sync_cuda();

  PointCloud _point_cloud;
};
} // namespace pc::devices
