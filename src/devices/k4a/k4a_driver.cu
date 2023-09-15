#include "k4a_driver.h"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

namespace pc::devices {

__host__ __device__ static inline float rad(float deg) {
  constexpr auto mult = 3.141592654f / 180.0f;
  return deg * mult;
};

// TODO these GPU kernels can probably be taken outside of the k4a classes and
// used with any sensor type

typedef thrust::tuple<short3, color, int> point_in_t;
typedef thrust::tuple<position, color> point_out_t;

struct point_transformer
    : public thrust::unary_function<point_in_t, point_out_t> {

  DeviceConfiguration config;
  Eigen::Vector3f alignment_center;
  Eigen::Vector3f aligned_position_offset;
  Eigen::Matrix3f auto_tilt_rotation;

  point_transformer(const DeviceConfiguration &device_config,
                    const position &aligned_center,
                    const position &position_offset,
                    const Eigen::Matrix3f &auto_tilt)
      : config(device_config), auto_tilt_rotation(auto_tilt) {
    alignment_center =
        Eigen::Vector3f(aligned_center.x, aligned_center.y, aligned_center.z);
    aligned_position_offset = Eigen::Vector3f(
        position_offset.x, position_offset.y, position_offset.z);
  }

  __device__ point_out_t operator()(point_in_t point) const {

    // we reinterpret our point in Eigen containers so we have easy maths
    using namespace Eigen;

    short3 pos = thrust::get<0>(point);

    // we put our position into a float vector because it allows us to
    // transform it by other float types (e.g. matrices, quaternions)
    Vector3f pos_f(pos.x, pos.y, pos.z);

    // perform any auto-tilt
    pos_f = auto_tilt_rotation * pos_f;

    // flip y and z axes for our world space
    pos_f = Vector3f(pos_f[0], -pos_f[1], -pos_f[2]);

    // create the rotation around our center
    AngleAxisf rot_x(rad(config.rotation_deg.x), Vector3f::UnitX());
    AngleAxisf rot_y(rad(config.rotation_deg.y), Vector3f::UnitY());
    AngleAxisf rot_z(rad(config.rotation_deg.z), Vector3f::UnitZ());
    Quaternionf q = rot_z * rot_y * rot_x;
    Affine3f rot_transform =
        Translation3f(-alignment_center) * q * Translation3f(alignment_center);

    // perform alignment transformation along with manual rotation
    pos_f =
        (rot_transform * pos_f) + alignment_center + aligned_position_offset;

    // perform our manual translation
    pos_f += Vector3f(config.offset.x, config.offset.y, config.offset.z);

    // specified axis flips
    if (config.flip_x)
      pos_f.x() = -pos_f.x();
    if (config.flip_y)
      pos_f.y() = -pos_f.y();
    if (config.flip_z)
      pos_f.z() = -pos_f.z();

    // and scaling
    pos_f *= config.scale;

    position pos_out = {(short)__float2int_rd(pos_f.x()),
                        (short)__float2int_rd(pos_f.y()),
                        (short)__float2int_rd(pos_f.z()), 0};

    color col = thrust::get<1>(point);
    // TODO apply color transformations here

    return thrust::make_tuple(pos_out, col);
  }
};

struct point_filter {
  DeviceConfiguration config;

  __device__ bool check_color(color value) const {
    // remove totally black values
    if (value.r == 0 && value.g == 0 && value.b == 0)
      return false;
    return true;
  }

  __device__ bool check_bounds(short3 value) const {
    if (value.x < config.crop_x.min)
      return false;
    if (value.x > config.crop_x.max)
      return false;
    if (value.y < config.crop_y.min)
      return false;
    if (value.y > config.crop_y.max)
      return false;
    if (value.z < config.crop_z.min)
      return false;
    if (value.z > config.crop_z.max)
      return false;
    return true;
  }

  __device__ bool sample(int index) const { return index % config.sample == 0; }

  __device__ bool operator()(point_in_t point) const {
    auto index = thrust::get<2>(point);
    if (!sample(index))
      return false;
    auto color = thrust::get<1>(point);
    if (!check_color(color))
      return false;
    auto position = thrust::get<0>(point);
    if (!check_bounds(position))
      return false;
    return true;
  }
};

using thrust_point_t = thrust::tuple<position, color>;

struct Translate {
  float x;
  float y;
  float z;

  __device__ thrust_point_t operator()(thrust_point_t point) const {
    auto position = thrust::get<0>(point);
    position.x += x;
    position.y += y;
    position.z += z;
    return thrust::make_tuple(position, thrust::get<1>(point));
  }
};

struct Scale {
  float x;
  float y;
  float z;

  __device__ thrust_point_t operator()(thrust_point_t point) const {
    auto position = thrust::get<0>(point);
    position.x *= x;
    position.y *= y;
    position.z *= z;
    return thrust::make_tuple(position, thrust::get<1>(point));
  }
};

template <typename Iterator, typename... Transformations>
auto make_transform_pipeline(Iterator begin, size_t count,
                             Transformations &&...transformations) {
  auto it = thrust::make_transform_iterator(
      begin, std::forward<Transformations>(transformations)...);
  auto it_end = thrust::make_transform_iterator(
      begin + count, std::forward<Transformations>(transformations)...);
  return std::make_pair<Iterator, Iterator>(it, it_end);
  // thrust::transform(it, it_end, begin,
  // std::forward<Transformations>(transformations)...);
}

PointCloud K4ADriver::point_cloud(const DeviceConfiguration &config) {

  if (!_buffers_updated || !is_open())
    return _point_cloud;

  _last_config = config;

  std::lock_guard<std::mutex> lock(_buffer_mutex);

  const uint positions_in_size = sizeof(short3) * incoming_point_count;
  const uint positions_out_size = sizeof(position) * incoming_point_count;
  const uint colors_size = sizeof(color) * incoming_point_count;

  // initialize our GPU memory
  short3 *incoming_positions;
  color *incoming_colors;
  short3 *filtered_positions;
  color *filtered_colors;
  position *output_positions;
  color *output_colors;

  cudaMallocManaged(&incoming_positions, positions_in_size);
  cudaMallocManaged(&incoming_colors, colors_size);
  cudaMallocManaged(&filtered_positions, positions_in_size);
  cudaMallocManaged(&filtered_colors, colors_size);
  cudaMallocManaged(&output_positions, positions_out_size);
  cudaMallocManaged(&output_colors, colors_size);

  // fill the GPU memory with our CPU buffers from the kinect
  cudaMemcpy(incoming_positions, _positions_buffer.data(), positions_in_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(incoming_colors, _colors_buffer.data(), colors_size,
             cudaMemcpyHostToDevice);

  // make some thrust::device_ptr from the raw pointers so they work
  // <algorithm> style
  thrust::device_ptr<short3> incoming_positions_ptr =
      thrust::device_pointer_cast(incoming_positions);
  thrust::device_ptr<color> incoming_colors_ptr =
      thrust::device_pointer_cast(incoming_colors);
  thrust::device_ptr<short3> filtered_positions_ptr =
      thrust::device_pointer_cast(filtered_positions);
  thrust::device_ptr<color> filtered_colors_ptr =
      thrust::device_pointer_cast(filtered_colors);
  thrust::device_ptr<position> output_positions_ptr =
      thrust::device_pointer_cast(output_positions);
  thrust::device_ptr<color> output_colors_ptr =
      thrust::device_pointer_cast(output_colors);

  // we create some sequence of point indices to zip so our GPU kernels have
  // access to them
  int *indices;
  cudaMallocManaged(&indices, positions_in_size);
  thrust::device_ptr<int> point_indices = thrust::device_pointer_cast(indices);
  thrust::sequence(point_indices, point_indices + incoming_point_count);

  // zip position and color buffers together so we can run our algorithms on
  // the dataset as a single point-cloud
  auto incoming_points_begin = thrust::make_zip_iterator(thrust::make_tuple(
      incoming_positions_ptr, incoming_colors_ptr, point_indices));
  auto incoming_points_end = thrust::make_zip_iterator(
      thrust::make_tuple(incoming_positions_ptr + incoming_point_count,
                         incoming_colors_ptr + incoming_point_count,
                         point_indices + incoming_point_count));
  auto filtered_points_begin = thrust::make_zip_iterator(thrust::make_tuple(
      filtered_positions_ptr, filtered_colors_ptr, point_indices));
  auto output_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(output_positions_ptr, output_colors_ptr));

  // run the kernels
  auto filtered_points_end =
      thrust::copy_if(incoming_points_begin, incoming_points_end,
                      filtered_points_begin, point_filter{config});
  thrust::transform(filtered_points_begin, filtered_points_end,
                    output_points_begin,
                    point_transformer(config, _alignment_center,
                                      _aligned_position_offset, _auto_tilt));

  // we can determine the output count using the resulting output iterator
  // from running the kernels
  auto output_point_count =
      std::distance(filtered_points_begin, filtered_points_end);

  // auto [it, it_end] = make_transform_pipeline(output_points_begin,
  // output_point_count,
  //                    Translate{1000, 1000, 1000});

  // TRANSLATE
  // auto [transform_start, transform_end] = make_transform_pipeline(
  //     output_points_begin, output_points_begin + output_point_count,
  //     Translate{5000, 5000, 5000});

  // thrust::transform(translate_start,
  // 		    translate_end,
  // 		    output_points_begin, Translate{-5000, -5000, -5000});

  // wait for the GPU process to complete
  cudaDeviceSynchronize();

  // copy back to our output point-cloud on the CPU
  const uint output_positions_size = sizeof(position) * output_point_count;
  const uint output_colors_size = sizeof(color) * output_point_count;
  _point_cloud.positions.resize(output_point_count);
  _point_cloud.colors.resize(output_point_count);
  cudaMemcpy(_point_cloud.positions.data(), output_positions,
             output_positions_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(_point_cloud.colors.data(), output_colors, output_colors_size,
             cudaMemcpyDeviceToHost);

  // clean up
  cudaFree(indices);
  cudaFree(incoming_positions);
  cudaFree(incoming_colors);
  cudaFree(filtered_positions);
  cudaFree(filtered_colors);
  cudaFree(output_positions);
  cudaFree(output_colors);

  _buffers_updated = false;

  return _point_cloud;
}

} // namespace pc::devices
