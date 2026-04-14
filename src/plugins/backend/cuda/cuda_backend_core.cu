#include "cuda_backend_core.h"

#include "../backend_utils.h"

#include <mutex>
#include <pointcaster/core_types.h>
#include <profiling/profiling_zone.h>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <unordered_map>

namespace {
// global data used by all instances of this backend

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
std::unordered_map<void *, DeviceTransformMemory> instance_device_memory;

void create_device_memory(void *owner, const size_t point_count) {
  DeviceTransformMemory new_device_memory{
      .point_count = point_count,
      .input_depth_data = thrust::device_vector<uint16_t>(point_count),
      .input_rgb_data = thrust::device_vector<color_rgb>(point_count),
      .output_positions = thrust::device_vector<position>(point_count),
      .output_colors = thrust::device_vector<color>(point_count),
      .interleaved_render = thrust::device_vector<std::byte>(point_count * 12),
      .indices = thrust::device_vector<int>(point_count)};

  thrust::sequence(new_device_memory.indices.begin(),
                   new_device_memory.indices.end());

  std::lock_guard lock(device_memory_access);
  instance_device_memory.emplace(owner, std::move(new_device_memory));
}

} // namespace

namespace pc::backend::cuda {

struct ProjectAndTransform {

  CameraIntrinsics camera_intrinsics;

  using OutputPointT = thrust::tuple<position, color>;
  using InputPixelT = thrust::tuple<uint16_t, color_rgb, int>;

  __host__ __device__ OutputPointT operator()(InputPixelT input) const {
    const uint16_t depth = thrust::get<0>(input);
    const color_rgb rgb = thrust::get<1>(input);
    const int i = thrust::get<2>(input);
    const auto &frame_width = camera_intrinsics.frame_width;

    const auto px = i % frame_width;
    const auto py = i / frame_width;

    // auto pos = util::project_2d_to_3d(px, py, depth, camera_intrinsics);
    position pos{static_cast<int16_t>(px), static_cast<int16_t>(py),
                 static_cast<int16_t>(depth)};

    color col{rgb.r, rgb.g, rgb.b};

    return thrust::make_tuple(pos, col);
  }
};

struct InterleaveForRender {
  const position *positions;
  const color *colors;
  std::byte *output;

  __host__ __device__ void operator()(int i) const {
    memcpy(output + i * 12, &positions[i], 8);
    memcpy(output + i * 12 + 8, &colors[i], 4);
  }
};

bool init_device_memory(void *owner, const size_t point_count) {
  try {
    create_device_memory(owner, point_count);
    return true;
  } catch (...) {
  }
  return false;
}

void free_device_memory(void *owner) {
  instance_device_memory.erase(owner);
}

void project_transform_frame_data(void *owner,
                                  UShortDepthData input_depth_frame,
                                  RgbColorData input_rgb_frame,
                                  std::shared_ptr<PointCloud> output_cloud,
                                  const CameraIntrinsics &camera_intrinsics,
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

  {
    ProfilingZone transform_zone("CudaBackend::transform");

    // TODO can/do these run in parallel?

    thrust::transform(thrust::cuda::par, frame_data_input_begin,
                      frame_data_input_end, output_points_begin,
                      ProjectAndTransform{camera_intrinsics});

    if (output_render_buffer) {
      thrust::for_each(
          thrust::cuda::par, device_memory->indices.begin(),
          device_memory->indices.end(),
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
                 device_memory->output_positions.end(),
                 output_cloud->positions.begin());
    thrust::copy(device_memory->output_colors.begin(),
                 device_memory->output_colors.end(),
                 output_cloud->colors.begin());

    if (output_render_buffer) {
      thrust::copy(device_memory->interleaved_render.begin(),
                   device_memory->interleaved_render.end(),
                   render_output.begin());
    }
  }
}

} // namespace pc::backend::cuda