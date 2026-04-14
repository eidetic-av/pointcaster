#include "cpu_backend.h"
#include "../backend_utils.h"
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <algorithm>
#include <core/logger/logger.h>
#include <execution>
#include <logger/logger.h>
#include <pointcaster/core_types.h>
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

void CpuBackend::project_transform_frame_data(
    UShortDepthData input_depth_frame, RgbColorData input_rgb_frame,
    std::shared_ptr<PointCloud> output_cloud,
    const CameraIntrinsics &color_intrinsics,
    const TransformConfiguration &transform,
    std::span<std::byte> render_output) {
  const auto point_count = output_cloud->size();
  const auto index_sequence =
      std::views::iota(0, static_cast<int>(point_count));

  const auto frame_width = color_intrinsics.frame_width;

  char *render_destination =
      render_output.empty() ? nullptr
                            : reinterpret_cast<char *>(render_output.data());

  // convert metres to milimetres for point-cloud space
  const pc::float3 translation{transform.position.x * 1000,
                               transform.position.y * 1000,
                               transform.position.z * 1000};

  const auto project_and_transform_point = [&](const auto i) {
    const auto &depth_pixel = input_depth_frame[i];
    const auto &color_pixel = input_rgb_frame[i];

    const auto px = i % frame_width;
    const auto py = i / frame_width;
    auto pos = util::project_2d_to_3d(px, py, depth_pixel, color_intrinsics);
    color col{color_pixel.r, color_pixel.g, color_pixel.b};

    // TODO
    // we add transform kernels here instead of inline transformation
    pos.x += translation.x;
    pos.y += translation.y;
    pos.z += translation.z;

    output_cloud->positions[i] = std::move(pos);
    output_cloud->colors[i] = std::move(col);

    if (render_destination) {
      std::memcpy(render_destination + i * 12, &pos, 8);
      std::memcpy(render_destination + i * 12 + 8, &col, 4);
    }
  };

  std::for_each(std::execution::par_unseq, index_sequence.begin(),
                index_sequence.end(), project_and_transform_point);
}

} // namespace pc::backend

CORRADE_PLUGIN_REGISTER(CpuBackend, pc::backend::CpuBackend,
                        "net.pointcaster.BackendPlugin/1.0")