#include "fringe_removal_operator.h"
#include "fringe_removal_config.h"
#include <algorithm>
#include <execution>
// #include <profiling/profiling_zone.h>

namespace pc::operators {

void FringeRemovalOperator::init() {
  pc::logger()->trace("Initialised FringeRemovalOperator");
}

void FringeRemovalOperator::process(
    const PointCloud &input, PointCloud &output,
    const OperatorConfigurationVariant &config_variant) const {
  // profiling::ProfilingZone process_zone("FringeRemovalOperator::process");
  const auto &config = std::get<FringeRemovalConfiguration>(config_variant);
  pc::logger()->debug("{} ({}::process)", config.id,
                      FringeRemovalConfiguration::PluginName);

  std::transform(std::execution::par_unseq, input.colors.begin(),
                 input.colors.end(), output.colors.begin(),
                 [](color c) { return color{.r = 255, .g = 0, .b = 0}; });
}

} // namespace pc::operators

CORRADE_PLUGIN_REGISTER(FringeRemovalOperator,
                        pc::operators::FringeRemovalOperator,
                        "net.pointcaster.OperatorPlugin/1.0")
