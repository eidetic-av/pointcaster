#include "../../logger.h"
#include "k4a_driver.h"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

namespace pc::devices {

typedef thrust::tuple<Short3, color, int> point_in_t;
typedef thrust::tuple<position, color> point_out_t;

__host__ __device__ static inline float as_rad(float deg) {
  constexpr auto mult = 3.141592654f / 180.0f;
  return deg * mult;
};

struct K4ADriverImplDeviceMemory {
  thrust::device_vector<Short3> incoming_positions;
  thrust::device_vector<color> incoming_colors;
  thrust::device_vector<Short3> filtered_positions;
  thrust::device_vector<color> filtered_colors;
  thrust::device_vector<position> output_positions;
  thrust::device_vector<color> output_colors;
  thrust::device_vector<int> indices;

  K4ADriverImplDeviceMemory(std::size_t point_count)
      : incoming_positions(point_count), incoming_colors(point_count),
        filtered_positions(point_count), filtered_colors(point_count),
        output_positions(point_count), output_colors(point_count),
        indices(point_count) {
    thrust::sequence(indices.begin(), indices.end());
  }
};

void K4ADriver::init_device_memory() {
  _device_memory = new K4ADriverImplDeviceMemory(incoming_point_count);
  _device_memory_ready = true;
  pc::logger->debug("Initialised device memory ({})", id());
}
void K4ADriver::free_device_memory() {
  _device_memory_ready = false;
  delete _device_memory;
  pc::logger->debug("Device memory freed ({})", id());
}

// TODO these GPU kernels can probably be taken outside of the k4a classes and
// used with any sensor type

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

    Short3 pos = thrust::get<0>(point);

    // we put our position into a float vector because it allows us to
    // transform it by other float types (e.g. matrices, quaternions)
    Vector3f pos_f(pos.x, pos.y, pos.z);

    // perform any auto-tilt
    pos_f = auto_tilt_rotation * pos_f;

    // flip y and z axes for our world space
    pos_f = Vector3f(pos_f[0], -pos_f[1], -pos_f[2]);

    // All K4A inputs seem to be rotated by ~7degrees amount for some reason...
    const AngleAxisf inbuilt_rot(as_rad(-7.0f), Vector3f::UnitX());

    // create the rotation around our center
    AngleAxisf rot_x(as_rad(config.rotation_deg.x), Vector3f::UnitX());
    AngleAxisf rot_y(as_rad(config.rotation_deg.y), Vector3f::UnitY());
    AngleAxisf rot_z(as_rad(config.rotation_deg.z), Vector3f::UnitZ());
    Quaternionf q = rot_z * rot_y * rot_x * inbuilt_rot;
    Affine3f rot_transform =
        Translation3f(-alignment_center) * q * Translation3f(alignment_center);

    // specified axis flips
    if (config.flip_x) pos_f.x() = -pos_f.x();
    if (config.flip_y) pos_f.y() = -pos_f.y();
    if (config.flip_z) pos_f.z() = -pos_f.z();

    // perform alignment transformation along with manual rotation
    pos_f =
        (rot_transform * pos_f) + alignment_center + aligned_position_offset;

    // perform our manual translation
    pos_f += Vector3f(config.offset.x, config.offset.y, config.offset.z);

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

  __device__ bool check_bounds(Short3 value) const {
    auto x = config.flip_x ? -value.x : value.x;
    auto y = config.flip_y ? -value.y : value.y;
    auto z = config.flip_z ? -value.z : value.z;
    return x >= config.crop_x.min && x <= config.crop_x.max &&
           y >= config.crop_y.min && y <= config.crop_y.max &&
           z >= config.crop_z.min && z <= config.crop_z.max;
  }

  __device__ bool sample(int index) const { return index % config.sample == 0; }

  __device__ bool operator()(point_in_t point) const {
    auto index = thrust::get<2>(point);
    if (!sample(index)) return false;
    auto color = thrust::get<1>(point);
    if (!check_color(color)) return false;
    auto position = thrust::get<0>(point);
    if (!check_bounds(position)) return false;
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

PointCloud K4ADriver::point_cloud(const DeviceConfiguration &config,
                                  OperatorList operator_list) {

  if (!_device_memory_ready || !_open || !_buffers_updated)
    return _point_cloud;

  _last_config = config;

  auto &incoming_positions = _device_memory->incoming_positions;
  auto &incoming_colors = _device_memory->incoming_colors;
  auto &filtered_positions = _device_memory->filtered_positions;
  auto &filtered_colors = _device_memory->filtered_colors;
  auto &output_positions = _device_memory->output_positions;
  auto &output_colors = _device_memory->output_colors;
  auto &indices = _device_memory->indices;

  // copy data from other threads
  std::unique_lock buffer_access(_buffer_mutex);

  thrust::copy(_positions_buffer.begin(), _positions_buffer.end(),
               incoming_positions.begin());
  thrust::copy(_colors_buffer.begin(), _colors_buffer.end(),
               incoming_colors.begin());

  // we can unlock our capture-thread buffers because we don't need them
  // again until after the GPU has finalised compute
  buffer_access.unlock();

  Eigen::Matrix3f auto_tilt_value;
  {
    std::lock_guard lock(_auto_tilt_value_mutex);
    auto_tilt_value = _auto_tilt_value;
  }

  // zip position and color buffers together so we can run our algorithms on
  // the dataset as a single point-cloud
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

  thrust::transform(
      filtered_points_begin, filtered_points_end, output_points_begin,
      point_transformer(config, _alignment_center, _aligned_position_offset,
                        auto_tilt_value));

  // we can determine the output count using the resulting output iterator
  // from running the kernels
  auto output_point_count =
      std::distance(filtered_points_begin, filtered_points_end);

  for (auto &operators : operator_list) {
    operators.get().run_operators(output_points_begin,
                                  output_points_begin + output_point_count);
  }

  // wait for the kernels to complete
  cudaDeviceSynchronize();

  // copy back to our output point-cloud on the CPU
  buffer_access.lock();

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
