#include "session.h"
#include "session/session_config.h"

#include <logger/logger.h>
#include <memory>
#include <pipeline/concurrent_operator_pipeline.h>
#include <plugins/devices/device_plugin.h>
#include <stop_token>
#include <workspace/workspace.h>

namespace pc {

Session::Session(Workspace &workspace, const SessionConfiguration &config)
    : id(config.id) {
  _workspace = &workspace;

  if (_workspace->backend_plugin_manager) {
    _cpu_backend =
        _workspace->backend_plugin_manager->instantiate("CpuBackend");
    if (_cpu_backend) _cpu_backend->init();
  }

  update_config(config);

  _update_thread = std::jthread(
      [this](std::stop_token stop) { update_loop(std::move(stop)); });
}

void Session::update_config(const SessionConfiguration &config) {
  _config = config;
  sync_operators(_config.operators);
}

std::vector<camera::CameraFrame> Session::latest_camera_frames() const {
  if (_pipeline) return _pipeline->latest_camera_frames();
  return {};
}

void Session::on_pipeline_output(std::shared_ptr<PointCloud> cloud) {
  if (cloud && !cloud->empty() && _cpu_backend) {
    auto buf = std::make_shared<std::vector<std::byte>>(cloud->size() * 16);
    _cpu_backend->pack_render_buffer(*cloud, *buf);
    _latest_render_data.store(std::move(buf), std::memory_order_release);
  }
  _current_point_cloud.store(cloud, std::memory_order_release);
  notify_point_cloud_updated();
}

void Session::update_loop(std::stop_token stop_token) {
  using namespace std::chrono;

  auto next_tick = steady_clock::now();

  std::vector<std::shared_ptr<PointCloud>> last_clouds;

  while (!stop_token.stop_requested()) {
    int update_hz = 90;

    std::vector<devices::DevicePlugin *> device_ptrs;
    {
      std::lock_guard lock(_workspace->config_access);
      auto &sessions = _workspace->config.sessions;
      auto it = std::ranges::find(sessions, id, &SessionConfiguration::id);
      if (it == sessions.end()) break; // session removed: exit loop
      update_hz = it->operator_pipeline.value().update_hz.value();
      for (auto &device : _workspace->devices) {
        if (device && !device->is_discovery_instance())
          device_ptrs.push_back(device.get());
      }
    }

    // snapshot current clouds
    std::vector<std::shared_ptr<PointCloud>> clouds;
    clouds.reserve(device_ptrs.size());
    for (auto *device : device_ptrs) clouds.push_back(device->point_cloud());

    // changed if the set of cloud identities differs from last tick.
    bool changed = clouds.size() != last_clouds.size();
    if (!changed) {
      for (size_t i = 0; i < clouds.size(); ++i) {
        if (clouds[i] != last_clouds[i]) {
          changed = true;
          break;
        }
      }
    }

    const bool force =
        _force_reprocess.exchange(false, std::memory_order_acq_rel);

    if (changed || force) {
      last_clouds = clouds;

      auto aggregated = std::make_shared<PointCloud>();
      for (auto &cloud : clouds) {
        if (!cloud || cloud->empty()) continue;
        aggregated->positions.insert(aggregated->positions.end(),
                                     cloud->positions.begin(),
                                     cloud->positions.end());
        aggregated->colors.insert(aggregated->colors.end(),
                                  cloud->colors.begin(), cloud->colors.end());
      }

      if (!aggregated->empty()) {
        feed_operator_pipeline(std::move(aggregated));
      }
      // (Optional) if aggregated is empty but we previously had data, you could
      // store an empty cloud + notify here to clear the session render when all
      // devices are removed.
    }

    const auto period = duration_cast<steady_clock::duration>(
        duration<double>(1.0 / std::max(update_hz, 1)));
    next_tick += period;
    next_tick = std::max(next_tick, steady_clock::now());
    std::this_thread::sleep_until(next_tick);
  }
}

} // namespace pc