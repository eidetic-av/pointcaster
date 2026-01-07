#pragma once

#include "../device_plugin.h"
#include "orbbec_device_config.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>

#include <atomic>
#include <chrono>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Context.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <pointcaster/point_cloud.h>
#include <print>
#include <profiling/profiling_mutex.h>
#include <thread>

namespace pc::devices {
// this device memory structure hides CUDA types, allowing OrbbecDriver to have
// CUDA members. This prevents issues when this header is included in TUs not
// compiled with nvcc
struct OrbbecImplDeviceMemory;

class OrbbecDevice final : public DevicePlugin {
public:
  explicit OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                        Corrade::Containers::StringView plugin);

  ~OrbbecDevice() override;

  OrbbecDevice(const OrbbecDevice &) = delete;
  OrbbecDevice &operator=(const OrbbecDevice &) = delete;
  OrbbecDevice(OrbbecDevice &&) = delete;
  OrbbecDevice &operator=(OrbbecDevice &&) = delete;

  DeviceStatus status() const override;

  // TODO: maybe it would better to return a constant reference to the
  // pointcloud here so its decided at the call site if a copy is required
  // pc::types::PointCloud point_cloud() const override;

  void start() override;
  void stop() override;
  void restart() override;

private:
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

  std::vector<OBColorPoint> _point_buffer;
  std::mutex _point_buffer_access;
  std::atomic_bool _buffer_updated{false};

  std::atomic_bool _running_pipeline = false;
  std::atomic<std::chrono::steady_clock::time_point> _last_updated_time{};
  std::atomic_bool _in_error_state{false};
  bool _initialised_point_cloud_scale = false;

  PC_PROFILING_MUTEX(_process_current_cloud_access);
  std::uint64_t _last_processed_frame_index{0};

  pc::types::PointCloud _current_point_cloud;

  OrbbecImplDeviceMemory *_device_memory;
  std::atomic_bool _device_memory_ready{false};

  bool init_device_memory(std::size_t incoming_point_count);
  void free_device_memory();

  void start_sync();
  void stop_sync();
  void restart_sync();

  void set_running(bool running_pipeline) {
    _running_pipeline = running_pipeline;
    notify_status_changed();
  }

  void set_error_state(bool error_state) {
    _in_error_state = error_state;
    notify_status_changed();
  }

  void set_updated_time(std::chrono::steady_clock::time_point new_time) {
    const auto old_time =
        _last_updated_time.exchange(new_time, std::memory_order_acq_rel);
    const bool new_invalid =
        (new_time == std::chrono::steady_clock::time_point{});
    const bool old_invalid =
        (old_time == std::chrono::steady_clock::time_point{});
    if (new_invalid != old_invalid) notify_status_changed();
  }

  void timeout_thread_work(std::stop_token stop_token);
};

} // namespace pc::devices
