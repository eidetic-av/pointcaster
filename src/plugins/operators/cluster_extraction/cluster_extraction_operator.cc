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

std::shared_ptr<PointCloud>
ClusterExtractionOperator::process(const PointCloud &input) {
  using profiling::ProfilingZone;
  auto variant = load_config();
  if (!variant) return std::make_shared<PointCloud>(input);

  const auto &config = std::get<ClusterExtractionConfiguration>(*variant);
  // any kid of lock needed here??
  pc::logger()->debug("find out how to set here: {}",
                      config.point_count.value());
  //   config.test_int.set(10);
  auto output = std::make_shared<PointCloud>(input);
  return output;
}

} // namespace pc::operators

CORRADE_PLUGIN_REGISTER(ClusterExtractionOperator,
                        pc::operators::ClusterExtractionOperator,
                        "net.pointcaster.OperatorPlugin/1.0")