#pragma once

#include "../device.h"
#include <libobsensor/ObSensor.hpp>
#include <thread>

namespace pc::devices {

// Forward declaration hides CUDA types, allowing OrbbecDriver to have CUDA
// members. This prevents issues when this header is included in TUs not
// compiled with nvcc.
struct OrbbecImplDeviceMemory;

class OrbbecDevice {

public:
  inline static std::vector<std::reference_wrapper<OrbbecDevice>>
      attached_devices;
  inline static std::mutex devices_access;

  OrbbecDevice(DeviceConfiguration &config, std::string_view id);
  ~OrbbecDevice();

  OrbbecDevice(const OrbbecDevice &) = delete;
  OrbbecDevice &operator=(const OrbbecDevice &) = delete;
  OrbbecDevice(OrbbecDevice &&) = delete;
  OrbbecDevice &operator=(OrbbecDevice &&) = delete;

  pc::types::PointCloud point_cloud(pc::operators::OperatorList operators = {});

  void draw_imgui_controls();

  // TODO this seems dumb
  DeviceConfiguration &config() { return _config; }
  const std::string& id() { return _id; }

private:
  DeviceConfiguration &_config;
  std::string _id;
  std::string _ip;

  std::unique_ptr<ob::Context> _ob_ctx;
  std::shared_ptr<ob::Pipeline> _ob_pipeline;

  std::unique_ptr<ob::PointCloudFilter> _ob_point_cloud;
  bool _initialised_point_cloud_scale = false;

  std::vector<OBColorPoint> _point_buffer;
  std::mutex _point_buffer_access;
  std::atomic_bool _buffer_updated;

  pc::types::PointCloud _current_point_cloud;

  OrbbecImplDeviceMemory *_device_memory;
  std::atomic_bool _device_memory_ready{false};
  void init_device_memory(std::size_t incoming_point_count);
  void free_device_memory();
};

} // namespace pc::devices
