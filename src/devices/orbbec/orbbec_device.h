#pragma once

#include "../device.h"
#include "orbbec_device_config.gen.h"
#include <atomic>
#include <libobsensor/ObSensor.hpp>
#include <optional>
#include <thread>
#include <chrono>


namespace pc::devices {

// Forward declaration hides CUDA types, allowing OrbbecDriver to have CUDA
// members. This prevents issues when this header is included in TUs not
// compiled with nvcc.
struct OrbbecImplDeviceMemory;

struct OrbbecDeviceInfo
{
    std::string ip;
    std::string serial_num;
};

class OrbbecDevice : public DeviceBase<OrbbecDeviceConfiguration> {

public:
  inline static std::vector<std::reference_wrapper<OrbbecDevice>>
      attached_devices;
  inline static std::mutex devices_access;
  inline static std::vector<OrbbecDeviceInfo> discovered_devices;
  inline static std::atomic_bool discovering_devices;

  static void init_context();
  static void destroy_context();

  static void discover_devices();

  explicit OrbbecDevice(OrbbecDeviceConfiguration &config);
  ~OrbbecDevice();

  OrbbecDevice(const OrbbecDevice &) = delete;
  OrbbecDevice &operator=(const OrbbecDevice &) = delete;
  OrbbecDevice(OrbbecDevice &&) = delete;
  OrbbecDevice &operator=(OrbbecDevice &&) = delete;

  DeviceStatus status() const override;
  pc::types::PointCloud point_cloud() override;

  void start();
  void stop();
  void restart();

  void start_sync();
  void stop_sync();
  void restart_sync();

  /// returns signal_detach bool, a request for if the device should be detached
  bool draw_imgui_controls();

  const std::string &ip() { return config().ip; }

private:
  inline static std::unique_ptr<ob::Context> _ob_ctx;
  inline static std::shared_ptr<ob::DeviceList> _ob_device_list;
  inline static std::mutex _start_stop_access;

  std::shared_ptr<ob::Config> _ob_config;
  std::shared_ptr<ob::Device> _ob_device;
  std::shared_ptr<ob::Pipeline> _ob_pipeline;
  std::unique_ptr<ob::PointCloudFilter> _ob_point_cloud;

  std::optional<OBXYTables> _ob_xy_tables;
  std::optional<std::vector<float>> _xy_table_data;
  std::optional<std::vector<OBPoint3f>> _depth_positions_buffer;
  std::optional<std::vector<OBColorPoint>> _depth_frame_buffer;

  std::jthread _initialisation_thread;
  std::jthread _timeout_thread;

  bool _running_pipeline = false;
  bool _initialised_point_cloud_scale = false;

  std::vector<OBColorPoint> _point_buffer;
  std::mutex _point_buffer_access;
  std::atomic_bool _buffer_updated;

  std::atomic<std::chrono::steady_clock::time_point> _last_updated_time{};
  std::atomic_bool _in_error_state = false;

  pc::types::PointCloud _current_point_cloud;

  OrbbecImplDeviceMemory *_device_memory;
  std::atomic_bool _device_memory_ready{false};

  bool init_device_memory(std::size_t incoming_point_count);
  void free_device_memory();
};

} // namespace pc::devices
