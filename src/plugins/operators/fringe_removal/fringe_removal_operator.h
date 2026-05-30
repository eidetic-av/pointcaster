#pragma once

#include "fringe_removal_config.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Pointer.h>
#include <Corrade/Containers/String.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <camera/camera_frame.h>
#include <camera/look_at_camera.h>
#include <optional>
#include <plugins/backend/backend_plugin.h>
#include <plugins/operators/operator_plugin.h>

namespace pc::operators {

class FringeRemovalOperator final : public OperatorPlugin {
public:
  explicit FringeRemovalOperator(
      Corrade::PluginManager::AbstractManager &manager,
      Corrade::Containers::StringView plugin)
      : OperatorPlugin(manager, plugin) {}

  ~FringeRemovalOperator() override {}

  FringeRemovalOperator(const FringeRemovalOperator &) = delete;
  FringeRemovalOperator &operator=(const FringeRemovalOperator &) = delete;
  FringeRemovalOperator(FringeRemovalOperator &&) = delete;
  FringeRemovalOperator &operator=(FringeRemovalOperator &&) = delete;

  void init(OperatorHost *host,
            Corrade::PluginManager::Manager<backend::BackendPlugin>
                &backend_plugin_manager) override;

  std::shared_ptr<PointCloud> process(const PointCloud &input) override;

  void on_config_field_changed(std::string_view path = "") override;

  std::vector<pc::camera::CameraFrameRef> camera_frames() override {
    std::vector<pc::camera::CameraFrameRef> frames;
    frames.push_back(std::ref(_input_image));
    frames.push_back(std::ref(_edge_detector_image));
    frames.push_back(std::ref(_drop_mask_image));
    return frames;
  }

  const FringeRemovalConfiguration &config() const {
    return std::get<FringeRemovalConfiguration>(config_variant());
  }

private:
  pc::camera::LookAtCamera _camera;

  pc::camera::CameraFrame _input_image;
  pc::camera::CameraFrame _edge_detector_image;
  pc::camera::CameraFrame _drop_mask_image;
};

} // namespace pc::operators