#include "cpu_backend.h"

#include "../backend_filters.h"
#include "../backend_utils.h"
#include <Corrade/PluginManager/AbstractManager.h>
#include <Corrade/PluginManager/AbstractPlugin.h>
#include <algorithm>
#include <core/logger/logger.h>
#include <execution>
#include <functional>
#include <logger/logger.h>
#include <numeric>
#include <pointcaster/core_types.h>
#include <ranges>

// TODO ensure TBB is linked and loaded
#include <oneapi/tbb/parallel_for.h>

namespace pc::backend {

namespace {

// we can accept inputs from different source frame types by having a transform
// function where the following PointSource concept is an argument. It's used as
// a way to convert any data type into the required core type (position&color)
// inside the parallelised hot path
template <typename F>
concept PointSource =
    std::invocable<F, int> &&
    std::same_as<std::invoke_result_t<F, int>, std::pair<position, color>>;

template <PointSource F>
void transform_from(F &&get_point, size_t point_count,
                    std::shared_ptr<PointCloud> output_cloud,
                    const TransformConfiguration &transform,
                    const ColorTransformConfiguration &color_transform,
                    std::span<std::byte> render_output) {

  // TODO a few of these per-transform allocations could move to cpu_backend
  // class members and be resized in init

  const auto transform_parameters =
      filter::TransformFilterParameters::from_config(transform,
                                                     color_transform);

  const auto sample_cloud = transform_parameters.sample > 1;

  auto index_sequence = std::views::iota(0, static_cast<int>(point_count));

  const auto transform_point = [&](const auto i) {
    if (sample_cloud && !filter::sample(i, transform_parameters)) {
      output_cloud->positions[i] = filter::invalid_position_value;
      return;
    }
    auto [position, color] = get_point(i);
    output_cloud->positions[i] =
        filter::transform(position, transform_parameters);
    output_cloud->colors[i] =
        filter::color_transform(color, transform_parameters);
  };

  std::for_each(std::execution::par_unseq, index_sequence.begin(),
                index_sequence.end(), transform_point);

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

} // namespace

CpuBackend::CpuBackend(Corrade::PluginManager::AbstractManager &manager,
                       Corrade::Containers::StringView plugin)
    : BackendPlugin(manager, plugin) {
  pc::logger()->trace("Initialised CPU backend");
};

CpuBackend::~CpuBackend() {
  pc::logger()->trace("Destroyed CPU backend");
};

void CpuBackend::transform_point_cloud(
    const PointCloud &input_cloud, std::shared_ptr<PointCloud> output_cloud,
    const TransformConfiguration &transform,
    const ColorTransformConfiguration &color_transform,
    std::span<std::byte> render_output) const {

  transform_from(
      [&](int i) -> std::pair<position, color> {
        return {input_cloud.positions[i], input_cloud.colors[i]};
      },
      output_cloud->size(), output_cloud, transform, color_transform,
      render_output);
}

void CpuBackend::transform_point_cloud(
    std::span<const position> input_positions,
    std::span<const color> input_colors,
    std::shared_ptr<PointCloud> output_cloud,
    const TransformConfiguration &transform,
    const ColorTransformConfiguration &color_transform,
    std::span<std::byte> render_output) const {

  transform_from(
      [&](int i) -> std::pair<position, color> {
        return {input_positions[i], input_colors[i]};
      },
      output_cloud->size(), output_cloud, transform, color_transform,
      render_output);
}

void CpuBackend::project_transform_frame_data(
    std::span<const uint16_t> input_depth_frame,
    std::span<const color_rgb> input_rgb_frame,
    std::shared_ptr<PointCloud> output_cloud,
    const CameraIntrinsics &color_intrinsics,
    const TransformConfiguration &transform,
    const ColorTransformConfiguration &color_transform,
    std::span<std::byte> render_output) const {

  const auto point_count = output_cloud->size();
  const auto frame_width = color_intrinsics.frame_width;

  const auto convert_point_data = [&](int i) -> std::pair<position, color> {
    const auto px = i % frame_width;
    const auto py = i / frame_width;
    const auto pos =
        util::project_2d_to_3d(px, py, input_depth_frame[i], color_intrinsics);
    const auto &rgb = input_rgb_frame[i];
    return {pos, color{rgb.r, rgb.g, rgb.b}};
  };

  transform_from(convert_point_data, point_count, output_cloud, transform,
                 color_transform, render_output);
}

} // namespace pc::backend

CORRADE_PLUGIN_REGISTER(CpuBackend, pc::backend::CpuBackend,
                        "net.pointcaster.BackendPlugin/1.0")