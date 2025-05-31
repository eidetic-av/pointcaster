#include "ply_sequence_player.h"

#include "../../logger.h"
#include "../../operators/operator.h"
#include "../../structs.h"
#include "../transform_filters.cuh"
#include "../../filters/chroma_key.cuh"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

namespace pc::devices {

using pc::types::PointCloud;

struct PlySequencePlayerImplDeviceMemory {
  size_t capacity;

  thrust::device_vector<position> incoming_positions;
  thrust::device_vector<color> incoming_colors;
  thrust::device_vector<indexed_point_t> incoming_point_data;
  thrust::device_vector<indexed_point_t> filtered_data;
  thrust::device_vector<position> transformed_positions;
  thrust::device_vector<color> transformed_colors;
  thrust::device_vector<position> output_positions;
  thrust::device_vector<color> output_colors;
  thrust::device_vector<int> indices;

  PlySequencePlayerImplDeviceMemory(std::size_t point_count)
      : capacity(point_count), incoming_positions(point_count),
        incoming_colors(point_count), incoming_point_data(point_count),
        filtered_data(point_count), transformed_positions(point_count),
        transformed_colors(point_count), output_positions(point_count),
        output_colors(point_count), indices(point_count) {
    thrust::sequence(indices.begin(), indices.end());
  }

  void ensure_capacity(size_t point_count) {
    if (point_count <= capacity) return;
    incoming_positions.resize(point_count);
    incoming_colors.resize(point_count);
    incoming_point_data.resize(point_count);
    filtered_data.resize(point_count);
    transformed_positions.resize(point_count);
    transformed_colors.resize(point_count);
    output_positions.resize(point_count);
    output_colors.resize(point_count);
    indices.resize(point_count);
    thrust::sequence(indices.begin(), indices.end());
    capacity = point_count;
  }
};

struct ply_to_input_points
    : public thrust::unary_function<thrust::tuple<position, color, int>, indexed_point_t> {
  __device__ indexed_point_t operator()(thrust::tuple<position, color, int> t) const {
    return thrust::make_tuple(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t));
  }
};

// fuse required input filters for this type into a single kernel
struct input_filter {
  input_transform_filter transform_filter;

  const bool chroma_key;
  filters::chroma_key chroma_key_filter;

  __host__ __device__ explicit input_filter(
      const PlySequencePlayerConfiguration &_config)
      : transform_filter{_config.transform},
        chroma_key(_config.color.chroma_key.enabled) {
    if (chroma_key) {
      const auto &ck = _config.color.chroma_key;
      chroma_key_filter = filters::chroma_key{
          ck.target_hue_degrees, ck.hue_width_degrees, ck.minimum_saturation,
          ck.minimum_value, ck.invert_mask};
    }
  }

  __device__ bool operator()(const indexed_point_t &p) const {
    bool keep_point = transform_filter(p);
    if (chroma_key && !chroma_key_filter(p)) keep_point = false;
    return keep_point;
  }
};

// same with transform filters

struct transform_filter {
  device_transform_filter dtf;
  color_transform_filter ctf;

  __host__ __device__ transform_filter(PlySequencePlayerConfiguration config)
      : dtf(config.transform), ctf(config.color) {};

  __device__ indexed_point_t operator()(const indexed_point_t &p) const {
    return ctf(dtf(p));
  }
};

bool PlySequencePlayer::init_device_memory(size_t point_count) {
  std::lock_guard lock(_device_memory_access);
  _device_memory_ready = false;
  if (point_count == 0) {
    pc::logger->warn("Can't initialise device memory with 0 sized point cloud");
    return false;
  }
  try {
    pc::logger->info("Max point count: {}", point_count);
    if (_device_memory != nullptr) {
      delete static_cast<PlySequencePlayerImplDeviceMemory *>(_device_memory);
      _device_memory = nullptr;
    }
    _device_memory = new PlySequencePlayerImplDeviceMemory(point_count);
    _device_memory_ready = true;
  } catch (std::exception &e) {
    pc::logger->error("Failed to initialise GPU memory for PlySequencePlayer");
    pc::logger->error("Exception: {}", e.what());
  }
  return _device_memory_ready;
}

void PlySequencePlayer::free_device_memory() {
  std::lock_guard lock(_device_memory_access);
  _device_memory_ready = false;
  if (_device_memory != nullptr) {
    delete static_cast<PlySequencePlayerImplDeviceMemory *>(_device_memory);
    _device_memory = nullptr;
  }
  pc::logger->debug("PlySequencePlayer GPU memory freed");
}

void PlySequencePlayer::ensure_device_memory_capacity(size_t point_count) {
  if (!_device_memory_ready || point_count == 0) return;
  std::lock_guard lock(_device_memory_access);
  _device_memory->ensure_capacity(point_count);
}

pc::types::PointCloud PlySequencePlayer::point_cloud() {
  if (!config().active) return {};
  
  pc::types::PointCloud cloud{};

  {
    if (!_device_memory_ready) { return _last_point_cloud; }
    if (!_buffer_initialised.load(std::memory_order_relaxed)) {
      return _last_point_cloud;
    }

    const size_t current_frame = _current_frame.load();
    const auto frame_index = std::min(_frame_buffer.size() - 1,
                                      current_frame % config().buffer_capacity);
    auto &frame = _frame_buffer[frame_index];
    if (frame.loaded.load() && frame.frame_index.load() == current_frame) {
      cloud = frame.point_cloud;
    } else {
      return _last_point_cloud;
    }
  }

  std::lock_guard gpu_lock(_device_memory_access);

  const auto point_count = cloud.size();
  _device_memory->ensure_capacity(point_count);

  auto &incoming_positions = _device_memory->incoming_positions;
  auto &incoming_colors = _device_memory->incoming_colors;
  auto &incoming_point_data = _device_memory->incoming_point_data;
  auto &filtered_data = _device_memory->filtered_data;
  auto &transformed_positions = _device_memory->transformed_positions;
  auto &transformed_colors = _device_memory->transformed_colors;
  auto &output_positions = _device_memory->output_positions;
  auto &output_colors = _device_memory->output_colors;
  auto &indices = _device_memory->indices;
  
  // copy host data into device memory.
  thrust::copy(cloud.positions.begin(), cloud.positions.end(),
               incoming_positions.begin());
  thrust::copy(cloud.colors.begin(), cloud.colors.end(),
               incoming_colors.begin());

  //combine positions, colours and indices
  auto ply_indexed_points_begin = thrust::make_zip_iterator(thrust::make_tuple(
      incoming_positions.begin(), incoming_colors.begin(), indices.begin()));

  // transform points into the format required form our transform filter kernels
  thrust::transform(ply_indexed_points_begin,
                    ply_indexed_points_begin + point_count,
                    incoming_point_data.begin(), ply_to_input_points());

  auto incoming_points_end = incoming_point_data.begin() + point_count;
  auto filtered_points_end =
      thrust::copy_if(incoming_point_data.begin(), incoming_points_end,
                      filtered_data.begin(), input_filter(config()));

  const auto filtered_point_count =
      thrust::distance(filtered_data.begin(), filtered_points_end);

  auto transformed_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(transformed_positions.begin(),
                         transformed_colors.begin(), indices.begin()));
  thrust::transform(filtered_data.begin(), filtered_points_end,
                    transformed_points_begin, transform_filter(config()));

  auto operator_output_begin = thrust::make_zip_iterator(thrust::make_tuple(
      output_positions.begin(), output_colors.begin(), indices.begin()));
  auto operator_output_end = thrust::copy_if(
      transformed_points_begin, transformed_points_begin + filtered_point_count,
      operator_output_begin, output_transform_filter{config().transform});

  // TODO: maybe we can make this cudaDeviceSynchronize call just set a flag or
  // something and then we only call a single cudaDeviceSynchronize() at the
  // very end of all Device point_cloud() calls ?
  cudaDeviceSynchronize();

  const auto output_point_count =
      std::distance(operator_output_begin, operator_output_end);
  cloud.positions.resize(output_point_count);
  cloud.colors.resize(output_point_count);

  thrust::copy(output_positions.begin(),
               output_positions.begin() + output_point_count,
               cloud.positions.begin());
  thrust::copy(output_colors.begin(),
               output_colors.begin() + output_point_count,
               cloud.colors.begin());

  _last_point_cloud = std::move(cloud);
  return _last_point_cloud;
}

} // namespace pc::devices