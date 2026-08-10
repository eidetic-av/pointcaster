#include "cluster_extraction_operator.h"

#include <profiling/profiling_zone.h>

namespace pc::operators {

void ClusterExtractionOperator::init(
    OperatorHost *host, Corrade::PluginManager::Manager<backend::BackendPlugin>
                            &backend_plugin_manager) {
  OperatorPlugin::init(host, backend_plugin_manager);
  pc::logger()->trace("Initialised ClusterExtractionOperator");
}

void ClusterExtractionOperator::on_config_field_changed(std::string_view path) {
  OperatorPlugin::on_config_field_changed(path);
  if (auto config_ptr = load_config()) {
    const auto config = std::get<ClusterExtractionConfiguration>(*config_ptr);
    // if (path.contains("camera")) {
    //   _camera.update_config(config.camera);
    // }
  }
}

PipelineFramePtr ClusterExtractionOperator::process(PipelineFramePtr input) {
  using profiling::ProfilingZone;

  auto variant = load_config();
  if (!variant) return input;
  const auto &config = std::get<ClusterExtractionConfiguration>(*variant);

  config.point_count.set(static_cast<int>(input->cloud->size()));

  return input;
}

} // namespace pc::operators

CORRADE_PLUGIN_REGISTER(ClusterExtractionOperator,
                        pc::operators::ClusterExtractionOperator,
                        "net.pointcaster.OperatorPlugin/1.0")
