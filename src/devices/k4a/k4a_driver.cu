#include "k4a_driver.h"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
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

PointCloud
K4ADriver::point_cloud(const DeviceConfiguration &config) {

  if (!_buffers_updated || !is_open())
    return _point_cloud;

  _last_config = config;

  // TODO all these vectors can be allocated once per K4ADriver instance
  // (but they can't be member variables because they are CUDA types and won't
  // compile in non-nvcc TUs that include k4a_driver.h)

  thrust::device_vector<short3> incoming_positions(incoming_point_count);
  thrust::device_vector<color> incoming_colors(incoming_point_count);
  thrust::device_vector<short3> filtered_positions(incoming_point_count);
  thrust::device_vector<color> filtered_colors(incoming_point_count);
  thrust::device_vector<position> output_positions(incoming_point_count);
  thrust::device_vector<color> output_colors(incoming_point_count);

  thrust::device_vector<int> indices(incoming_point_count);
  thrust::sequence(indices.begin(), indices.end());

  std::lock_guard<std::mutex> lock(_buffer_mutex);

  thrust::copy(_positions_buffer.begin(), _positions_buffer.end(),
               incoming_positions.begin());
  thrust::copy(_colors_buffer.begin(), _colors_buffer.end(),
               incoming_colors.begin());

  // // zip position and color buffers together so we can run our algorithms on
  // // the dataset as a single point-cloud
  auto incoming_points_begin = thrust::make_zip_iterator(thrust::make_tuple(
      incoming_positions.begin(), incoming_colors.begin(), indices.begin()));
  auto incoming_points_end = thrust::make_zip_iterator(thrust::make_tuple(
      incoming_positions.end(), incoming_colors.end(), indices.end()));

  auto filtered_points_begin = thrust::make_zip_iterator(thrust::make_tuple(
      filtered_positions.begin(), filtered_colors.begin(), indices.begin()));

  auto output_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(output_positions.begin(), output_colors.begin()));

  // run the kernels
  auto filtered_points_end =
      thrust::copy_if(incoming_points_begin, incoming_points_end,
		      filtered_points_begin, point_filter{config});
  thrust::transform(filtered_points_begin, filtered_points_end,
		    output_points_begin,
		    point_transformer(config, _alignment_center,
				      _aligned_position_offset, _auto_tilt));

  // wait for the kernels to complete
  cudaDeviceSynchronize();

  // we can determine the output count using the resulting output iterator
  // from running the kernels
  auto output_point_count =
      std::distance(filtered_points_begin, filtered_points_end);

  // copy back to our output point-cloud on the CPU
  auto output_positions_size = sizeof(position) * output_point_count;
  auto output_colors_size = sizeof(color) * output_point_count;
  _point_cloud.positions.resize(output_point_count);
  _point_cloud.colors.resize(output_point_count);

  thrust::copy(output_positions.begin(),
	       output_positions.begin() + output_point_count,
	       _point_cloud.positions.begin());
  thrust::copy(output_colors.begin(),
	       output_colors.begin() + output_point_count,
	       _point_cloud.colors.begin());

  _buffers_updated = false;

  return _point_cloud;
}

} // namespace pc::devices
