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

  // TODO placeholder...

  auto clusters = std::make_shared<AabbList>();
  clusters->resize(2);
  const auto drift = static_cast<int16_t>(input->seq % 500);
  clusters->min_positions()[0] = {-500, -500, drift};
  clusters->max_positions()[0] = {500, 500, static_cast<int16_t>(drift + 1000)};
  clusters->min_positions()[1] = {static_cast<int16_t>(-1500 + drift), -500,
                                  -500};
  clusters->max_positions()[1] = {static_cast<int16_t>(-500 + drift), 500, 500};

  // send the output data downstream the pipeline for other operators
  // that need access, they can get to it accessing the "clusters" stream
  auto output_frame = input->clone();
  output_frame->set_stream("clusters", clusters);

  // and set the output value on the config, so that publishers can watch that
  // through the config registry and react to updates
  config.clusters.set(std::move(clusters));

  return output_frame;
}

} // namespace pc::operators

CORRADE_PLUGIN_REGISTER(ClusterExtractionOperator,
                        pc::operators::ClusterExtractionOperator,
                        "net.pointcaster.OperatorPlugin/1.0")
