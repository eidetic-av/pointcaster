#include "ply_sequence_player.h"

#include "../../logger.h"
#include "../../operators/operator.h"
#include "../../structs.h"
#include "../transform_filters.cuh"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

namespace pc::devices {

using pc::types::PointCloud;

struct PlySequencePlayerImplDeviceMemory {
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
      : incoming_positions(point_count), incoming_colors(point_count),
        incoming_point_data(point_count), filtered_data(point_count),
        transformed_positions(point_count), transformed_colors(point_count),
        output_positions(point_count), output_colors(point_count),
        indices(point_count)
  {
    thrust::sequence(indices.begin(), indices.end());
  }
};

struct ply_to_input_points
    : public thrust::unary_function<thrust::tuple<position, color, int>, indexed_point_t> {
  __device__ indexed_point_t operator()(thrust::tuple<position, color, int> t) const {
    return thrust::make_tuple(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t));
  }
};

bool PlySequencePlayer::init_device_memory(size_t point_count) {
  _device_memory_ready = false;
  try {
    pc::logger->info("Max point count: {}", point_count);
    _device_memory = new PlySequencePlayerImplDeviceMemory(point_count);
    _device_memory_ready = true;
  } catch (std::exception &e) {
    pc::logger->error("Failed to initialise GPU memory for PlySequencePlayer");
    pc::logger->error("Exception: {}", e.what());
  }
  return _device_memory_ready;
}

void PlySequencePlayer::free_device_memory() {
  _device_memory_ready = false;
  if (_device_memory != nullptr) {
    delete static_cast<PlySequencePlayerImplDeviceMemory *>(_device_memory);
    _device_memory = nullptr;
  }
  pc::logger->debug("PlySequencePlayer GPU memory freed");
}

pc::types::PointCloud PlySequencePlayer::point_cloud() {
  if (!config().active) return {};

  if (!_device_memory_ready || !_buffer_updated) {
    return _current_point_cloud;
  }

  const auto frame_index =
      std::min(current_frame(), _pointcloud_buffer.size() - 1);
  pc::types::PointCloud cloud = _pointcloud_buffer.at(frame_index);

  if (cloud.positions.empty()) return cloud;
  
  auto &incoming_positions = _device_memory->incoming_positions;
  auto &incoming_colors = _device_memory->incoming_colors;
  auto &incoming_point_data = _device_memory->incoming_point_data;
  auto &filtered_data = _device_memory->filtered_data;
  auto &transformed_positions = _device_memory->transformed_positions;
  auto &transformed_colors = _device_memory->transformed_colors;
  auto &output_positions = _device_memory->output_positions;
  auto &output_colors = _device_memory->output_colors;
  auto &indices = _device_memory->indices;
  
  const size_t point_count = cloud.positions.size();
  
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
  auto filtered_points_end = thrust::copy_if(
      incoming_point_data.begin(), incoming_points_end, filtered_data.begin(),
      input_transform_filter(config().transform));

  const auto filtered_point_count =
      thrust::distance(filtered_data.begin(), filtered_points_end);

  auto transformed_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(transformed_positions.begin(),
                         transformed_colors.begin(), indices.begin()));
  thrust::transform(filtered_data.begin(), filtered_points_end,
                    transformed_points_begin,
                    device_transform_filter(config().transform));

  auto operator_output_begin = thrust::make_zip_iterator(thrust::make_tuple(
      output_positions.begin(), output_colors.begin(), indices.begin()));
  auto operator_output_end = thrust::copy_if(
      transformed_points_begin, transformed_points_begin + filtered_point_count,
      operator_output_begin, output_transform_filter{config().transform});

  cudaDeviceSynchronize();

  const auto output_point_count =
      std::distance(operator_output_begin, operator_output_end);
  _current_point_cloud.positions.resize(output_point_count);
  _current_point_cloud.colors.resize(output_point_count);

  thrust::copy(output_positions.begin(),
               output_positions.begin() + output_point_count,
               _current_point_cloud.positions.begin());
  thrust::copy(output_colors.begin(),
               output_colors.begin() + output_point_count,
               _current_point_cloud.colors.begin());

  _buffer_updated = false;

  return _current_point_cloud;
}

} // namespace pc::devices