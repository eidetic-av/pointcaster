#include "cpu_backend.h"
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <core/logger/logger.h>

#include <algorithm>
#include <execution>
#include <logger/logger.h>
#include <ranges>

// TODO ensure TBB is linked and loaded
#include <oneapi/tbb/parallel_for.h>

namespace pc::backend {

CpuBackend::CpuBackend(Corrade::PluginManager::AbstractManager &manager,
                       Corrade::Containers::StringView plugin)
    : BackendPlugin(manager, plugin) {
  pc::logger()->trace("Initialised CPU backend");
};

CpuBackend::~CpuBackend() {
  pc::logger()->trace("Destroyed CPU backend");
};

void CpuBackend::transform_point_cloud(
    const TransformConfiguration &transform, PointCloud &output_cloud,
    const PointGeneratorFunction &generate_point) {
  const auto point_count = output_cloud.size();
  const auto index_sequence =
      std::views::iota(0, static_cast<int>(point_count));

  // TODO i guess this lambda might contain globally defined transform kernels
  // that work on both cpu and gpu

  const auto transform_point = [&](const auto index) {
    auto [pos, col] = generate_point(index);
    // TODO
    // do transform stuff here
    output_cloud.positions[index] = std::move(pos);
    output_cloud.colors[index] = std::move(col);
  };

  std::for_each(std::execution::par_unseq, index_sequence.begin(),
                index_sequence.end(), transform_point);
};

} // namespace pc::backend

CORRADE_PLUGIN_REGISTER(CpuBackend, pc::backend::CpuBackend,
                        "net.pointcaster.BackendPlugin/1.0")