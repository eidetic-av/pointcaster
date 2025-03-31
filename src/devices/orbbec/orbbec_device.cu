#include "../../logger.h"
#include "../transform_filters.cuh"
#include "orbbec_device.h"
#include <Eigen/Eigen>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>


namespace pc::devices {

using pc::types::color;
using pc::types::position;

typedef thrust::tuple<OBColorPoint, int> ob_point_in_t;

struct OrbbecImplDeviceMemory {
  thrust::device_vector<OBColorPoint> incoming_data;
  thrust::device_vector<indexed_point_t> incoming_point_data;
  thrust::device_vector<indexed_point_t> filtered_data;
  thrust::device_vector<position> transformed_positions;
  thrust::device_vector<color> transformed_colors;
  thrust::device_vector<position> output_positions;
  thrust::device_vector<color> output_colors;
  thrust::device_vector<int> indices;

  OrbbecImplDeviceMemory(std::size_t point_count)
      : incoming_data(point_count), incoming_point_data(point_count),
        filtered_data(point_count), transformed_positions(point_count),
        transformed_colors(point_count), output_positions(point_count),
        output_colors(point_count), indices(point_count) {
    thrust::sequence(indices.begin(), indices.end());
  }
};

struct ob_to_input_points
    : public thrust::unary_function<ob_point_in_t, indexed_point_t> {
  __device__ indexed_point_t
  operator()(thrust::tuple<OBColorPoint, int> t) const {
    OBColorPoint p = thrust::get<0>(t);
    int idx = thrust::get<1>(t);
    position pos{(float)p.x, (float)p.y, (float)p.z, 0.0f};
    color col{(uint8_t)p.b, (uint8_t)p.g, (uint8_t)p.r};
    return thrust::make_tuple(pos, col, idx);
  }
};

bool OrbbecDevice::init_device_memory(std::size_t incoming_point_count) {
  _device_memory_ready = false;
  try {
    _device_memory = new OrbbecImplDeviceMemory(incoming_point_count);
    _device_memory_ready = true;
  } catch (std::exception e) {
    pc::logger->error("Failed to initialise GPU memory");
    pc::logger->error("Exception: {}", e.what());
  }
  return _device_memory_ready;
}
void OrbbecDevice::free_device_memory() {
  _device_memory_ready = false;
  if (_device_memory != nullptr) delete _device_memory;
  pc::logger->debug("OrbbecDevice GPU memory freed ({})", _ip);
}

pc::types::PointCloud
OrbbecDevice::point_cloud(pc::operators::OperatorList operators) {

  if (!config().active) return {};

  if (!_device_memory_ready || !_buffer_updated) {
    return _current_point_cloud;
  }

  auto &incoming_data = _device_memory->incoming_data;
  auto &incoming_point_data = _device_memory->incoming_point_data;
  auto &filtered_data = _device_memory->filtered_data;
  auto &indices = _device_memory->indices;
  auto &transformed_positions = _device_memory->transformed_positions;
  auto &transformed_colors = _device_memory->transformed_colors;
  auto &output_positions = _device_memory->output_positions;
  auto &output_colors = _device_memory->output_colors;

  auto& transform_config = config().transform;

  size_t incoming_point_count;

  // copy data from other threads
  {
    std::lock_guard lock(_point_buffer_access);

    thrust::copy(_point_buffer.begin(), _point_buffer.end(),
		 incoming_data.begin());

    incoming_point_count = _point_buffer.size();
  }

  auto ob_indexed_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(incoming_data.begin(), indices.begin()));

  auto ob_indexed_points_end = ob_indexed_points_begin + incoming_point_count;

  thrust::transform(ob_indexed_points_begin, ob_indexed_points_end,
                    incoming_point_data.begin(), ob_to_input_points{});

  auto incoming_points_end = incoming_point_data.begin() + incoming_point_count;

  auto filtered_points_end = thrust::copy_if(
      incoming_point_data.begin(), incoming_points_end, filtered_data.begin(),
      input_transform_filter{transform_config});

  auto filtered_point_count =
      thrust::distance(filtered_data.begin(), filtered_points_end);

  // transform the filtered points, placing them into transformed_points
  auto transformed_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(transformed_positions.begin(),
			 transformed_colors.begin(), indices.begin()));

  thrust::transform(filtered_data.begin(), filtered_points_end,
                    transformed_points_begin,
                    device_transform_filter(transform_config));

  auto operator_output_begin = thrust::make_zip_iterator(thrust::make_tuple(
      output_positions.begin(), output_colors.begin(), indices.begin()));

  // copy transformed_points into operator_output if they pass the output_filter
  auto operator_output_end = thrust::copy_if(
      transformed_points_begin, transformed_points_begin + filtered_point_count,
      operator_output_begin, output_transform_filter{transform_config});

  for (auto &operator_host_ref : operators) {
    auto &operator_host = operator_host_ref.get();
    operator_output_end = pc::operators::SessionOperatorHost::run_operators(
        operator_output_begin, operator_output_end, operator_host._config);
  }

  // wait for the kernels to complete
  cudaDeviceSynchronize();

  // we can determine the output count using the resulting output iterator
  // from running the kernels
  auto output_point_count =
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
