#include "cpu_backend.h"

#include "../backend_filters.h"
#include "../backend_utils.h"
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <algorithm>
#include <core/logger/logger.h>
#include <execution>
#include <logger/logger.h>
#include <numeric>
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

// TODO a few of these per-transform allocations could move to cpu_backend class
// members and be resized in init

void CpuBackend::project_transform_frame_data(
    UShortDepthData input_depth_frame, RgbColorData input_rgb_frame,
    std::shared_ptr<PointCloud> output_cloud,
    const CameraIntrinsics &color_intrinsics,
    const TransformConfiguration &transform,
    const ColorTransformConfiguration &color_transform,
    std::span<std::byte> render_output) {

  const auto point_count = output_cloud->size();
  const auto frame_width = color_intrinsics.frame_width;

  const auto transform_parameters =
      filter::TransformFilterParameters::from_config(transform,
                                                     color_transform);

  const auto sample_cloud = transform_parameters.sample > 1;

  auto index_sequence = std::views::iota(0, static_cast<int>(point_count));

  const auto project_and_transform_point = [&](const auto i) {
    if (sample_cloud && !filter::sample(i, transform_parameters)) {
      output_cloud->positions[i] = filter::invalid_position_value;
      return;
    }
    const auto px = i % frame_width;
    const auto py = i / frame_width;
    auto position =
        util::project_2d_to_3d(px, py, input_depth_frame[i], color_intrinsics);
    position = filter::transform(position, transform_parameters);

    auto color =
        filter::color_transform(input_rgb_frame[i], transform_parameters);

    output_cloud->positions[i] = position;
    output_cloud->colors[i] = color;
  };

  std::for_each(std::execution::par_unseq, index_sequence.begin(),
                index_sequence.end(), project_and_transform_point);

  std::vector<int> output_indices(point_count);
  std::iota(output_indices.begin(), output_indices.end(), 0);

  const auto crop_point_to_bounds = [&](const auto i) {
    return filter::is_valid(output_cloud->positions[i]) &&
           filter::in_bounds(output_cloud->positions[i], transform_parameters);
  };

  auto new_end =
      std::partition(std::execution::par_unseq, output_indices.begin(),
                     output_indices.end(), crop_point_to_bounds);

  const size_t new_point_count = std::distance(output_indices.begin(), new_end);

  std::vector<position> output_positions(new_point_count);
  std::vector<color> output_colors(new_point_count);

  auto output_range = std::views::iota(size_t{0}, new_point_count);

  char *render_destination =
      render_output.empty() ? nullptr
                            : reinterpret_cast<char *>(render_output.data());

  const auto copy_to_output_buffers = [&](const auto i) {
    const auto output_index = output_indices[i];
    output_positions[i] = output_cloud->positions[output_index];
    output_colors[i] = output_cloud->colors[output_index];

    if (render_destination) {
      // for our shader we pack the positions and the index too
      // which allows us to do point cloud lookup using screen-space color
      // picking
      std::memcpy(render_destination + i * 16, &output_positions[i], 8);
      std::memcpy(render_destination + i * 16 + 8, &output_colors[i], 4);
      float idx = static_cast<float>(i);
      std::memcpy(render_destination + i * 16 + 12, &idx, 4);
    }
  };

  std::for_each(std::execution::par_unseq, output_range.begin(),
                output_range.end(), copy_to_output_buffers);

  output_cloud->bounds =
      std::transform_reduce(std::execution::par_unseq, output_positions.begin(),
                            output_positions.end(), position_bounds{},
                            filter::merge_bounds, filter::as_bounds);

  std::copy(output_positions.begin(), output_positions.end(),
            output_cloud->positions.begin());
  std::copy(output_colors.begin(), output_colors.end(),
            output_cloud->colors.begin());

  output_cloud->resize(new_point_count);
}

} // namespace pc::backend

CORRADE_PLUGIN_REGISTER(CpuBackend, pc::backend::CpuBackend,
                        "net.pointcaster.BackendPlugin/1.0")