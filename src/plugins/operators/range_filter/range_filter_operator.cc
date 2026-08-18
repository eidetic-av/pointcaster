#include "range_filter_operator.h"

#include <profiling/profiling_zone.h>

namespace pc::operators {

void RangeFilterOperator::init(
    OperatorHost *host, Corrade::PluginManager::Manager<backend::BackendPlugin>
                            &backend_plugin_manager) {
  OperatorPlugin::init(host, backend_plugin_manager);
  pc::logger()->trace("Initialised RangeFilterOperator");
}

PipelineFramePtr RangeFilterOperator::process(PipelineFramePtr input) {
  using profiling::ProfilingZone;

  auto variant = load_config();
  if (!variant) return input;
  const auto &config = std::get<RangeFilterConfiguration>(*variant);

  ProfilingZone operator_zone("RangeFilterOperator");
  operator_zone.text(config.id);

  // TODO

  return input;
}

} // namespace pc::operators

CORRADE_PLUGIN_REGISTER(RangeFilterOperator, pc::operators::RangeFilterOperator,
                        "net.pointcaster.OperatorPlugin/1.0")
