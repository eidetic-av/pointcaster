#include "orbbec_device.h"
#include "../../logger.h"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <Eigen/Eigen>

namespace pc::devices {

using pc::types::color;
using pc::types::position;

typedef thrust::tuple<OBColorPoint, int> ob_point_in_t;
typedef thrust::tuple<position, color, int> indexed_point_t;

struct OrbbecImplDeviceMemory {
  thrust::device_vector<OBColorPoint> incoming_data;
  thrust::device_vector<OBColorPoint> filtered_data;
  thrust::device_vector<position> transformed_positions;
  thrust::device_vector<color> transformed_colors;
  thrust::device_vector<position> output_positions;
  thrust::device_vector<color> output_colors;
  thrust::device_vector<int> indices;

  OrbbecImplDeviceMemory(std::size_t point_count)
      : incoming_data(point_count), filtered_data(point_count),
	transformed_positions(point_count), transformed_colors(point_count),
        output_positions(point_count), output_colors(point_count),
        indices(point_count) {
    thrust::sequence(indices.begin(), indices.end());
  }
};

__host__ __device__ static inline float as_rad(float deg) {
  constexpr auto mult = 3.141592654f / 180.0f;
  return deg * mult;
};

struct input_filter {
  DeviceTransformConfiguration config;

  __device__ bool check_empty(OBColorPoint value) const {
    if (value.x == 0.0f && value.y == 0.0f && value.z == 0.0f)
      return false;
    if (value.r == 0.0f && value.g == 0.0f && value.b == 0.0f)
      return false;
    return true;
  }

  __device__ bool check_crop(OBColorPoint value) const {
    auto x = config.flip_x ? -value.x : value.x;
    auto y = config.flip_y ? value.y : -value.y;
    auto z = config.flip_z ? -value.z : value.z;
    return x >= config.crop_x.min && x <= config.crop_x.max &&
           y >= config.crop_y.min && y <= config.crop_y.max &&
           z >= config.crop_z.min && z <= config.crop_z.max;
  }

  __device__ bool sample(int index) const {
    return (index % config.sample) == 0;
  }

  __device__ bool operator()(ob_point_in_t point) const {
    auto index = thrust::get<1>(point);
    if (!sample(index)) return false;
    auto point_data = thrust::get<0>(point);
    if (!check_empty(point_data)) return false;
    if (!check_crop(point_data)) return false;
    return true;
  }
};

struct point_transformer
    : public thrust::unary_function<ob_point_in_t, indexed_point_t> {

  DeviceTransformConfiguration config;
  Eigen::Vector3f alignment_center;
  Eigen::Vector3f aligned_position_offset;
  Eigen::Matrix3f auto_tilt_rotation;

  point_transformer(const DeviceTransformConfiguration &transform_config,
                    const position &aligned_center,
                    const position &position_offset,
                    const Eigen::Matrix3f &auto_tilt)
      : config(transform_config), auto_tilt_rotation(auto_tilt) {
    alignment_center =
        Eigen::Vector3f(aligned_center.x, aligned_center.y, aligned_center.z);
    aligned_position_offset = Eigen::Vector3f(
        position_offset.x, position_offset.y, position_offset.z);
  }

  __device__ indexed_point_t operator()(ob_point_in_t point) const {

    // we reinterpret our point in Eigen containers so we have easy maths
    using namespace Eigen;

    OBColorPoint p = thrust::get<0>(point);

    Vector3f pos_f(p.x, p.y, p.z);

    Vector3f flip(config.flip_x ? -1 : 1,
		  config.flip_y ? -1 : 1,
		  config.flip_z ? -1 : 1);

    // TODO:
    // // perform any auto-tilt
    // pos_f = auto_tilt_rotation * pos_f;

    // flip y and z axes for our world space
    pos_f = Vector3f(pos_f[0], -pos_f[1], -pos_f[2]);

    // All K4A inputs seem to be rotated by ~7degrees amount for some reason...
    // const AngleAxisf inbuilt_rot(as_rad(-7.0f), Vector3f::UnitX());

    // input translation
    pos_f = pos_f + Vector3f{ config.translate.x * flip.x(),
			      config.translate.y * flip.y(),
			      config.translate.z * flip.z() };


    // create the rotation around our center
    AngleAxisf rot_x(as_rad(config.rotation_deg.x), Vector3f::UnitX());
    AngleAxisf rot_y(as_rad(-config.rotation_deg.y), Vector3f::UnitY());
    AngleAxisf rot_z(as_rad(config.rotation_deg.z), Vector3f::UnitZ());
    Quaternionf q = rot_z * rot_y * rot_x;
    Affine3f rot_transform =
      Translation3f(-alignment_center) * q * Translation3f(alignment_center);

    // specified axis flips
    pos_f = { pos_f.x() * flip.x(), pos_f.y() * flip.y(),
	      pos_f.z() * flip.z() };

    // perform alignment transformation along with manual rotation
    pos_f =
        (rot_transform * pos_f) + alignment_center + aligned_position_offset;

    // perform our alignment offset translation
    pos_f += Vector3f(config.offset.x, config.offset.y, config.offset.z);

    // and scaling
    pos_f *= config.scale;

    position pos_out = {(short)__float2int_rd(pos_f.x()),
                        (short)__float2int_rd(pos_f.y()),
                        (short)__float2int_rd(pos_f.z()), 0};

    int index = thrust::get<1>(point);

    pc::types::color col{(uint8_t)__float2uint_rn(p.b),
			 (uint8_t)__float2uint_rn(p.g),
			 (uint8_t)__float2uint_rn(p.r)};

    return thrust::make_tuple(pos_out, col, index);
  }
};

struct output_filter {
  DeviceTransformConfiguration config;

  __device__ bool check_bounds(position value) const {

    auto x = config.flip_x ? -value.x : value.x;
    auto y = config.flip_y ? -value.y : value.y;
    auto z = config.flip_z ? -value.z : value.z;

    return x >= config.bound_x.min && x <= config.bound_x.max &&
           y >= config.bound_y.min && y <= config.bound_y.max &&
           z >= config.bound_z.min && z <= config.bound_z.max;
  }

  __device__ bool operator()(indexed_point_t point) const {
    auto pos = thrust::get<0>(point);
    return check_bounds(pos);
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
  
  if (!_device_memory_ready || !_buffer_updated) {
    return _current_point_cloud;
  }

  auto &incoming_data = _device_memory->incoming_data;
  auto &filtered_data = _device_memory->filtered_data;
  auto &indices = _device_memory->indices;
  auto &transformed_positions = _device_memory->transformed_positions;
  auto &transformed_colors = _device_memory->transformed_colors;
  auto &output_positions = _device_memory->output_positions;
  auto &output_colors = _device_memory->output_colors;

  auto& transform_config = _config.transform;

  // copy data from other threads
  {
    std::lock_guard lock(_point_buffer_access);

    thrust::copy(_point_buffer.begin(), _point_buffer.end(),
		 incoming_data.begin());
  }

  // zip together the points and their indices
  auto incoming_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(incoming_data.begin(), indices.begin()));
  auto incoming_points_end = thrust::make_zip_iterator(
      thrust::make_tuple(incoming_data.end(), indices.end()));

  auto filtered_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(filtered_data.begin(), indices.begin()));

  // copy incoming_points into filtered_points if they pass they input_filter
  auto filtered_points_end =
      thrust::copy_if(incoming_points_begin, incoming_points_end,
		      filtered_points_begin, input_filter{transform_config});

  auto filtered_point_count =
      thrust::distance(filtered_points_begin, filtered_points_end);


  // transform the filtered points, placing them into transformed_points
  auto transformed_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(transformed_positions.begin(),
			 transformed_colors.begin(), indices.begin()));

  // TODO:
  position _alignment_center{};
  position _aligned_position_offset{};
  Eigen::Quaternion<float> _aligned_orientation_offset{};
  Eigen::Matrix3f auto_tilt_value;


  thrust::transform(
      filtered_points_begin, filtered_points_end, transformed_points_begin,
      point_transformer(transform_config, _alignment_center, _aligned_position_offset,
			auto_tilt_value));

  auto operator_output_begin = thrust::make_zip_iterator(thrust::make_tuple(
      output_positions.begin(), output_colors.begin(), indices.begin()));

  // copy transformed_points into operator_output if they pass the output_filter
  auto operator_output_end = thrust::copy_if(
      transformed_points_begin, transformed_points_begin + filtered_point_count,
      operator_output_begin, output_filter{transform_config});

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
