#include "workspace.h"

#include "point_streamer/point_streamer.h"
#include "recorder/session_recorder.h"
#include "session/session.h"

#include <chrono>
#include <cstddef>
#include <metrics/metrics.h>
#include <plugins/devices/device_plugin.h>
#include <plugins/devices/device_status.h>
#include <plugins/devices/device_variants.h>
#include <pointcaster/task_pool.h>
#include <string>
#include <thread>
#include <vector>

namespace pc {

namespace {

using LabelPairView = metrics::PrometheusServer::LabelPairView;

constexpr std::string_view status_name(devices::DeviceStatus status) {
  switch (status) {
  case devices::DeviceStatus::Unloaded:
    return "unloaded";
  case devices::DeviceStatus::Loading:
    return "loading";
  case devices::DeviceStatus::Loaded:
    return "loaded";
  case devices::DeviceStatus::Active:
    return "active";
  case devices::DeviceStatus::Missing:
    return "missing";
  }
  return "unknown";
}

constexpr auto every_status = {
    devices::DeviceStatus::Unloaded, devices::DeviceStatus::Loading,
    devices::DeviceStatus::Loaded, devices::DeviceStatus::Active,
    devices::DeviceStatus::Missing};

struct DeviceSample {
  std::string id;
  std::string plugin;
  devices::DeviceStatus status;
  std::size_t frames_processed;
  std::size_t dropped_frames;
  std::size_t frame_tasks_in_flight;
  std::size_t point_count;
};

} // namespace

void Workspace::metrics_thread_work(std::stop_token stop_token) {
  using namespace std::chrono_literals;

  while (!stop_token.stop_requested()) {
    std::this_thread::sleep_for(1s);

    if (!metrics::PrometheusServer::is_enabled()) continue;

    std::vector<DeviceSample> device_samples;
    {
      std::scoped_lock lock(config_access);
      device_samples.reserve(devices.size());
      for (auto &device : devices) {
        if (!device || device->is_discovery_instance()) continue;
        auto [device_id, plugin_name] =
            devices::device_info_from_variant(device->config_variant());
        const auto cloud = device->point_cloud();
        device_samples.push_back(
            {.id = std::move(device_id),
             .plugin = std::move(plugin_name),
             .status = device->status(),
             .frames_processed = device->frames_processed(),
             .dropped_frames = device->dropped_frames(),
             .frame_tasks_in_flight = device->frame_tasks_in_flight(),
             .point_count = cloud ? cloud->size() : 0});
      }
    }

    metrics::set_gauge("pointcaster_devices", device_samples.size());

    for (const auto &device : device_samples) {
      const std::initializer_list<LabelPairView> labels{
          {"device_id", device.id}, {"plugin", device.plugin}};

      metrics::set_counter("pointcaster_device_frames_processed_total",
                           device.frames_processed, labels);
      metrics::set_counter("pointcaster_device_frames_dropped_total",
                           device.dropped_frames, labels);
      metrics::set_gauge("pointcaster_device_frame_tasks_in_flight",
                         device.frame_tasks_in_flight, labels);
      metrics::set_gauge("pointcaster_device_max_frame_tasks",
                         devices::DevicePlugin::max_frame_tasks_in_flight,
                         labels);
      metrics::set_gauge("pointcaster_device_points", device.point_count,
                         labels);

      for (const auto status : every_status) {
        metrics::set_gauge("pointcaster_device_status",
                           device.status == status ? 1 : 0,
                           {{"device_id", device.id},
                            {"plugin", device.plugin},
                            {"status", status_name(status)}});
      }
    }

    metrics::set_gauge("pointcaster_sessions", sessions.size());

    for (const auto &[session_id, session] : sessions) {
      if (!session) continue;
      const auto cloud = session->point_cloud();
      metrics::set_gauge("pointcaster_session_points",
                         cloud ? cloud->size() : 0,
                         {{"session_id", session_id}});
      metrics::set_gauge("pointcaster_session_playing",
                         session->playback_is_playing() ? 1 : 0,
                         {{"session_id", session_id}});
      metrics::set_gauge("pointcaster_session_playback_frame",
                         session->playback_current_frame(),
                         {{"session_id", session_id}});
    }

    if (session_recorder) {
      metrics::set_gauge("pointcaster_recorder_recording",
                         session_recorder->is_recording() ? 1 : 0);
      metrics::set_gauge("pointcaster_recorder_writing_files",
                         session_recorder->is_file_writing() ? 1 : 0);
      metrics::set_gauge("pointcaster_recorder_writer_queue_depth",
                         session_recorder->writer_queue_depth());
      metrics::set_gauge("pointcaster_recorder_frame",
                         session_recorder->current_recording_frame());
      metrics::set_counter("pointcaster_recorder_frames_dropped_total",
                           session_recorder->dropped_frames());
    }

    if (point_streamer) {
      if (const auto counts = point_streamer->subscriber_counts()) {
        for (const auto &[channel, count] : *counts) {
          metrics::set_gauge("pointcaster_stream_subscribers", count,
                             {{"channel", channel}});
        }
      }
    }

    auto &pool = pc::task_pool();
    metrics::set_gauge("pointcaster_task_pool_threads",
                       pool.get_thread_count());
    metrics::set_gauge("pointcaster_task_pool_tasks_queued",
                       pool.get_tasks_queued());
    metrics::set_gauge("pointcaster_task_pool_tasks_running",
                       pool.get_tasks_running());
  }
}

} // namespace pc
