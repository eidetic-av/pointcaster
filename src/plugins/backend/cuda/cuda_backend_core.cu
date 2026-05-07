#include "cuda_backend_core.h"

#include "../backend_filters.h"
#include "../backend_utils.h"

#include <config/transform_config.h>
#include <mutex>
#include <pointcaster/core_types.h>
#include <profiling/profiling_zone.h>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <unordered_map>

namespace {

using namespace pc;

struct DeviceTransformMemory {
  size_t point_count;
  thrust::device_vector<uint16_t> input_depth_data;
  thrust::device_vector<color_rgb> input_rgb_data;
  thrust::device_vector<position> output_positions;
  thrust::device_vector<color> output_colors;
  thrust::device_vector<std::byte> interleaved_render;
  thrust::device_vector<int> indices;
};

std::mutex device_memory_access;
std::unordered_map<const void *, DeviceTransformMemory> instance_device_memory;

void create_device_memory(const void *owner, const size_t point_count) {
  DeviceTransformMemory new_device_memory{
      .point_count = point_count,
      .input_depth_data = thrust::device_vector<uint16_t>(point_count),
      .input_rgb_data = thrust::device_vector<color_rgb>(point_count),
      .output_positions = thrust::device_vector<position>(point_count),
      .output_colors = thrust::device_vector<color>(point_count),
      .interleaved_render = thrust::device_vector<std::byte>(point_count * 16),
      .indices = thrust::device_vector<int>(point_count)};

  thrust::sequence(new_device_memory.indices.begin(),
                   new_device_memory.indices.end());

  std::lock_guard lock(device_memory_access);
  instance_device_memory.emplace(owner, std::move(new_device_memory));
}

} // namespace

namespace pc::backend::cuda {

struct ProjectAndTransform {

  CameraIntrinsics color_intrinsics;
  filter::TransformFilterParameters params;
  bool sample_cloud;

  explicit ProjectAndTransform(
      const CameraIntrinsics &intrinsics,
      const TransformConfiguration &transform,
      const ColorTransformConfiguration &color_transform)
      : color_intrinsics(intrinsics) {
    params = filter::TransformFilterParameters::from_config(transform,
                                                            color_transform);
    sample_cloud = params.sample > 1;
  }

  using OutputPointT = thrust::tuple<position, color>;
  using InputPixelT = thrust::tuple<uint16_t, color_rgb, int>;

  __host__ __device__ OutputPointT operator()(InputPixelT input) const {
    const int i = thrust::get<2>(input);
    if (sample_cloud && !filter::sample(i, params)) {
      return thrust::make_tuple(filter::invalid_position_value, color{});
    }
    const uint16_t depth = thrust::get<0>(input);
    const color_rgb rgb = thrust::get<1>(input);
    const auto &frame_width = color_intrinsics.frame_width;

    const auto px = i % frame_width;
    const auto py = i / frame_width;

    auto position = util::project_2d_to_3d(px, py, depth, color_intrinsics);
    position = filter::transform(position, params);

    auto col = filter::color_transform(color{rgb.r, rgb.g, rgb.b}, params);

    return thrust::make_tuple(position, col);
  }
};

struct BoundsCheck {
  filter::TransformFilterParameters params;

  __host__ __device__ bool
  operator()(thrust::tuple<position, color> point) const {
    auto pos = thrust::get<0>(point);
    return filter::is_valid(pos) && filter::in_bounds(pos, params);
  }
};

struct InterleaveForRender {
  const position *positions;
  const color *colors;
  std::byte *output;

  __host__ __device__ void operator()(int i) const {
    memcpy(output + i * 16, &positions[i], 8);
    memcpy(output + i * 16 + 8, &colors[i], 4);
    float idx = static_cast<float>(i);
    memcpy(output + i * 16 + 12, &idx, 4);
  }
};

struct PositionAsBounds {
  __host__ __device__ position_bounds operator()(const position &p) const {
    return filter::as_bounds(p);
  }
};

struct MergeBounds {
  __host__ __device__ position_bounds
  operator()(const position_bounds &a, const position_bounds &b) const {
    return filter::merge_bounds(a, b);
  }
};

bool init_device_memory(const void *owner, const size_t point_count) {
  try {
    create_device_memory(owner, point_count);
    return true;
  } catch (...) {
  }
  return false;
}

void free_device_memory(const void *owner) {
  instance_device_memory.erase(owner);
}

void project_transform_frame_data(
    const void *owner, std::span<const uint16_t> input_depth_frame,
    std::span<const color_rgb> input_rgb_frame,
    std::shared_ptr<PointCloud> output_cloud,
    const CameraIntrinsics &color_intrinsics,
    const TransformConfiguration &transform,
    const ColorTransformConfiguration &color_transform,
    std::span<std::byte> render_output) {

  DeviceTransformMemory *device_memory;
  {
    std::lock_guard lock(device_memory_access);
    auto it = instance_device_memory.find(owner);
    if (it == instance_device_memory.end())
      throw std::runtime_error("device memory is not initialised");
    device_memory = &it->second;
  }

  using namespace pc::profiling;

  const auto transform_parameters =
      filter::TransformFilterParameters::from_config(transform,
                                                     color_transform);

  bool output_render_buffer = !render_output.empty();

  {
    ProfilingZone copy_zone("CudaBackend::copy_to_device");
    thrust::copy(input_depth_frame.begin(), input_depth_frame.end(),
                 device_memory->input_depth_data.begin());
    thrust::copy(input_rgb_frame.begin(), input_rgb_frame.end(),
                 device_memory->input_rgb_data.begin());
  }

  // zip together our input depth, colour and indices for use inside a single
  // algorithm
  auto frame_data_input_begin = thrust::make_zip_iterator(thrust::make_tuple(
      device_memory->input_depth_data.begin(),
      device_memory->input_rgb_data.begin(), device_memory->indices.begin()));
  auto frame_data_input_end =
      frame_data_input_begin + device_memory->point_count;

  // similarly zip together our output destination memory so the SoA output is
  // available in the single kernel
  auto output_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(device_memory->output_positions.begin(),
                         device_memory->output_colors.begin()));

  size_t new_point_count;
  position_bounds new_cloud_bounds;

  {
    ProfilingZone transform_zone("CudaBackend::transform_and_filter");

    thrust::transform(
        thrust::cuda::par, frame_data_input_begin, frame_data_input_end,
        output_points_begin,
        ProjectAndTransform{color_intrinsics, transform, color_transform});

    auto output_points_end = output_points_begin + device_memory->point_count;

    auto new_end =
        thrust::partition(thrust::cuda::par, output_points_begin,
                          output_points_end, BoundsCheck{transform_parameters});

    new_point_count = new_end - output_points_begin;

    new_cloud_bounds = thrust::transform_reduce(
        thrust::cuda::par, device_memory->output_positions.begin(),
        device_memory->output_positions.begin() + new_point_count,
        PositionAsBounds{}, position_bounds{}, MergeBounds{});

    if (output_render_buffer) {
      // reset indices to sequential for the filtered range
      thrust::sequence(device_memory->indices.begin(),
                       device_memory->indices.begin() + new_point_count);

      thrust::for_each(
          thrust::cuda::par, device_memory->indices.begin(),
          device_memory->indices.begin() + new_point_count,
          InterleaveForRender{
              thrust::raw_pointer_cast(device_memory->output_positions.data()),
              thrust::raw_pointer_cast(device_memory->output_colors.data()),
              thrust::raw_pointer_cast(
                  device_memory->interleaved_render.data())});
    }
  }

  {
    ProfilingZone output_zone("CudaBackend::copy_back_to_host");

    thrust::copy(device_memory->output_positions.begin(),
                 device_memory->output_positions.begin() + new_point_count,
                 output_cloud->positions.begin());
    thrust::copy(device_memory->output_colors.begin(),
                 device_memory->output_colors.begin() + new_point_count,
                 output_cloud->colors.begin());

    output_cloud->bounds = new_cloud_bounds;

    if (output_render_buffer) {
      thrust::copy(device_memory->interleaved_render.begin(),
                   device_memory->interleaved_render.begin() +
                       new_point_count * 16,
                   render_output.begin());
    }
  }

  output_cloud->resize(new_point_count);
}

} // namespace pc::backend::cuda