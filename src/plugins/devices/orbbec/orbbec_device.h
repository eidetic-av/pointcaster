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
#include <core/logger/logger.h>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Context.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <pointcaster/point_cloud.h>
#include <readerwriterqueue/readerwritercircularbuffer.h>
#include <thread>

// TODO profiling mutex not working if tracy is disabled on plugin startup
// #include <profiling/profiling_mutex.h>

namespace pc::devices {

class OrbbecDevice final : public DevicePlugin {
public:
  explicit OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                        Corrade::Containers::StringView plugin);

  ~OrbbecDevice() override;

  OrbbecDevice(const OrbbecDevice &) = delete;
  OrbbecDevice &operator=(const OrbbecDevice &) = delete;
  OrbbecDevice(OrbbecDevice &&) = delete;
  OrbbecDevice &operator=(OrbbecDevice &&) = delete;

  void init(Workspace &workspace) override;

  std::vector<DiscoveredDevice> discovered_devices() const override;
  void refresh_discovery() override;
  void add_discovery_change_callback(std::function<void()> cb) override;
  bool has_discovery_change_callback() const override;

  DeviceStatus status() const override;

  const PointCloud &point_cloud() override;

  void start() override;
  void stop() override;
  void restart() override;

  void on_config_field_changed(std::string_view path = "") override;

private:

    // TODO replace with SPMC triple-buffer
  moodycamel::BlockingReaderWriterCircularBuffer<PointCloud> _frame_buffer{2};

  PointCloud _current_point_cloud{{}, {}};

  std::vector<OBColorPoint> _point_buffer;
  std::mutex _point_buffer_access;
  std::atomic_bool _buffer_updated{false};

  std::atomic_bool _running_pipeline = false;
  std::atomic_bool _loading_pipeline = false;
  std::atomic_bool _in_error_state{false};
  bool _initialised_point_cloud_scale = false;

  std::atomic<std::chrono::steady_clock::time_point> _last_updated_time{};
  std::atomic<float> _pipeline_fps_ema{0.0f};
  std::atomic<std::chrono::steady_clock::time_point> _pipeline_last_tick{
      std::chrono::steady_clock::time_point{}};

  // TODO profiling mutex not working if tracy is disabled on plugin startup
  // PC_PROFILING_MUTEX(_process_current_cloud_access);
  std::mutex _process_current_cloud_access;
  std::uint64_t _last_processed_frame_index{0};

  std::jthread _initialisation_thread;
  std::jthread _pipeline_thread;
  std::jthread _timeout_thread;

  void start_sync();
  void stop_sync();

  void set_ip(std::string_view ip_address, std::string_view subnet_mask,
              std::string_view gateway_address);

  void set_running(bool running_pipeline) {
    _running_pipeline = running_pipeline;
    notify_status_changed();
  }

  void set_loading(bool loading_pipeline) {
    _loading_pipeline = loading_pipeline;
    notify_status_changed();
  }

  void set_error_state(bool error_state) {
    _in_error_state = error_state;
    notify_status_changed();
  }

  void set_updated_time(std::chrono::steady_clock::time_point new_time);

  void pipeline_thread_work(std::stop_token stop_token,
                            std::shared_ptr<ob::Device> ob_device);
  void timeout_thread_work(std::stop_token stop_token);
};

} // namespace pc::devices
