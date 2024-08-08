#pragma once

#include "../device.h"
#include <libobsensor/ObSensor.hpp>
#include <thread>
#include <atomic>

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

class OrbbecDevice {

public:
  inline static std::vector<std::reference_wrapper<OrbbecDevice>>
      attached_devices;
  inline static std::mutex devices_access;
  inline static std::vector<OrbbecDeviceInfo> discovered_devices;
  inline static std::atomic_bool discovering_devices;

  static void init_context();
  static void discover_devices();

  OrbbecDevice(DeviceConfiguration &config);
  ~OrbbecDevice();

  OrbbecDevice(const OrbbecDevice &) = delete;
  OrbbecDevice &operator=(const OrbbecDevice &) = delete;
  OrbbecDevice(OrbbecDevice &&) = delete;
  OrbbecDevice &operator=(OrbbecDevice &&) = delete;

  pc::types::PointCloud point_cloud(pc::operators::OperatorList operators = {});

  void start();
  void stop();
  void restart();

  void draw_imgui_controls();

  DeviceConfiguration &config() { return _config; }
  const std::string& ip() { return _ip; }

private:
  inline static std::unique_ptr<ob::Context> _ob_ctx;
  inline static std::shared_ptr<ob::DeviceList> _ob_device_list;

  DeviceConfiguration &_config;
  std::string _ip;

  std::shared_ptr<ob::Config> _ob_config;
  std::shared_ptr<ob::Device> _ob_device;
  std::shared_ptr<ob::Pipeline> _ob_pipeline;
  std::unique_ptr<ob::PointCloudFilter> _ob_point_cloud;

  std::jthread _initialisation_thread;

  bool _running_pipeline = false;
  bool _initialised_point_cloud_scale = false;

  std::vector<OBColorPoint> _point_buffer;
  std::mutex _point_buffer_access;
  std::atomic_bool _buffer_updated;

  pc::types::PointCloud _current_point_cloud;

  OrbbecImplDeviceMemory *_device_memory;
  std::atomic_bool _device_memory_ready{false};

  void start_pipeline();
  void stop_pipeline();

  void init_device_memory(std::size_t incoming_point_count);
  void free_device_memory();
};

} // namespace pc::devices
