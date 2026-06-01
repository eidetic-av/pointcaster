#pragma once

#include "session_config.h"

#include <Corrade/Containers/Pointer.h>
#include <atomic>
#include <camera/camera_frame.h>
#include <functional>
#include <logger/logger.h>
#include <memory>
#include <plugins/backend/backend_plugin.h>
#include <plugins/operators/operator_host.h>
#include <pointcaster/point_cloud.h>
#include <thread>

namespace pc {

class Workspace;

namespace operators {
class OperatorPlugin;
}

class Session : public operators::OperatorHost {
public:
  const std::string id;

  explicit Session(Workspace &workspace, const SessionConfiguration &config);

  void reprocess() override {
    _force_reprocess.store(true, std::memory_order_release);
  }

  void set_point_cloud_updated_callback(std::function<void()> cb) {
    _point_cloud_updated_callback = std::move(cb);
  }

  std::shared_ptr<PointCloud> point_cloud() const {
    return _current_point_cloud.load(std::memory_order_acquire);
  }

  std::shared_ptr<std::vector<std::byte>> render_data() const {
    return _latest_render_data.load(std::memory_order_acquire);
  }

  void update_config(const SessionConfiguration &config);

  std::vector<camera::CameraFrame> latest_camera_frames() const;

protected:
  void on_pipeline_output(std::shared_ptr<PointCloud> cloud) override;

  pipeline::ConcurrentOperatorPipelineConfiguration &
  pipeline_config() override {
    return _config.operator_pipeline.value();
  }

private:
  SessionConfiguration _config;
  std::jthread _update_thread;

  Corrade::Containers::Pointer<backend::BackendPlugin> _cpu_backend;

  std::atomic<std::shared_ptr<PointCloud>> _current_point_cloud{nullptr};
  std::atomic<std::shared_ptr<std::vector<std::byte>>> _latest_render_data{
      nullptr};
  std::function<void()> _point_cloud_updated_callback;

  std::atomic<bool> _force_reprocess{false};

  void notify_point_cloud_updated() {
    if (_point_cloud_updated_callback) _point_cloud_updated_callback();
  }

  void update_loop(std::stop_token stop_token);
};

} // namespace pc