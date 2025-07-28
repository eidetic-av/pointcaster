#include "../../logger.h"
#include "../../profiling.h"
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
    position pos{(short)p.x, (short)p.y, (short)p.z, 0};
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
  pc::logger->debug("OrbbecDevice GPU memory freed ({})", config().ip);
}

pc::types::PointCloud OrbbecDevice::point_cloud() {

  using namespace pc::profiling;
  ProfilingZone zone("OrbbecDevice::point_cloud");

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
  auto &transform_config = config().transform;
  size_t incoming_point_count;

  {
    ProfilingZone copy_zone("Copy Point Buffer");

    std::lock_guard lock(_point_buffer_access);
    thrust::copy(_point_buffer.begin(), _point_buffer.end(),
                 incoming_data.begin());
    incoming_point_count = _point_buffer.size();

    copy_zone.text(std::format("PointCount: {}", incoming_point_count));
  }

  size_t filtered_point_count;
  {
    ProfilingZone input_transform_zone("Transform & Filter Input Points");

    auto ob_indexed_points_begin = thrust::make_zip_iterator(
        thrust::make_tuple(incoming_data.begin(), indices.begin()));
    thrust::transform(ob_indexed_points_begin,
                      ob_indexed_points_begin + incoming_point_count,
                      incoming_point_data.begin(), ob_to_input_points{});
    auto incoming_points_end =
        incoming_point_data.begin() + incoming_point_count;
    auto filtered_points_end = thrust::copy_if(
        incoming_point_data.begin(), incoming_points_end, filtered_data.begin(),
        input_transform_filter{transform_config});
    filtered_point_count =
        thrust::distance(filtered_data.begin(), filtered_points_end);

    input_transform_zone.text(
        std::format("FilteredCount: {}", filtered_point_count));
  }

  auto filtered_begin = filtered_data.begin();
  auto filtered_end = filtered_begin + filtered_point_count;

  {
    ProfilingZone device_transform_zone("Device Transform Filter");
    auto out_zip = thrust::make_zip_iterator(
        thrust::make_tuple(transformed_positions.begin(),
                           transformed_colors.begin(), indices.begin()));
    thrust::transform(filtered_begin, filtered_end, out_zip,
                      device_transform_filter{transform_config});
  }

  size_t output_point_count;
  {
    ProfilingZone output_filter_zone("Apply Output Filter & Operators");
    auto in_zip = thrust::make_zip_iterator(
        thrust::make_tuple(transformed_positions.begin(),
                           transformed_colors.begin(), indices.begin()));
    auto in_end = in_zip + filtered_point_count;
    auto out_zip = thrust::make_zip_iterator(thrust::make_tuple(
        output_positions.begin(), output_colors.begin(), indices.begin()));

    auto transform_output_end = thrust::copy_if(
        in_zip, in_end, out_zip, output_transform_filter{transform_config});

    output_point_count = thrust::distance(out_zip, transform_output_end);
  }

  {
    ProfilingZone host_copy_zone("GPU to CPU copy");
    cudaDeviceSynchronize();
    _current_point_cloud.positions.resize(output_point_count);
    _current_point_cloud.colors.resize(output_point_count);

    thrust::copy(output_positions.begin(),
                 output_positions.begin() + output_point_count,
                 _current_point_cloud.positions.begin());
    thrust::copy(output_colors.begin(),
                 output_colors.begin() + output_point_count,
                 _current_point_cloud.colors.begin());
    host_copy_zone.text(std::format("OutputCount: {}", output_point_count));
  }

  _buffer_updated = false;
  return _current_point_cloud;
}

} // namespace pc::devices
