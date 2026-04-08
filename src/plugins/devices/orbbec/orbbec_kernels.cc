#include "orbbec_kernels.h"
#include "plugins/devices/orbbec/orbbec_device_config.h"

#include <algorithm>
#include <execution>
#include <iostream>
#include <libobsensor/h/ObTypes.h>
#include <libobsensor/hpp/Utils.hpp>
#include <logger/logger.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <ranges>
#include <span>

namespace pc::devices::orbbec {

void transform(const uint16_t *depth_frame_data_ptr,
               const color_rgb *color_frame_data_ptr, PointCloud &output_cloud,
               const OrbbecDeviceConfiguration &device_config,
               const OBCalibrationParam &calibration_parameters) {

  const auto point_count = output_cloud.size();
  std::span depth_data{depth_frame_data_ptr, point_count};
  std::span color_data{color_frame_data_ptr, point_count};

  const auto color_intrinsic =
      calibration_parameters.intrinsics[OB_SENSOR_COLOR];

  // we dont need an extrinsic for the 2d->3d conversion because frame alignment has already
  // been performed within orbbec sdk
  constexpr static OBExtrinsic identity_extrinsic{
      {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
      {0.0f, 0.0f, 0.0f},
  };

  const auto frame_width =
      device_config.color_resolution ==
              OrbbecDeviceConfiguration::ColorResolution::FHD_1920x1080
          ? 1920
          : 1280;

  const auto transform_point = [&](const auto i) {
    const auto x = i % frame_width;
    const auto y = i / frame_width;

    const auto ob_depth = depth_data[i];
    const auto ob_color = color_data[i];

    OBPoint3f result;

    ob::CoordinateTransformHelper::transformation2dto3d(
        OBPoint2f(x, y), ob_depth, color_intrinsic, identity_extrinsic,
        &result);

    output_cloud.positions[i] = {
        .x = static_cast<int16_t>(result.x),
        .y = -static_cast<int16_t>(result.y),
        .z = -static_cast<int16_t>(result.z)
    };

    output_cloud.colors[i] = {
        .r = static_cast<uint8_t>(ob_color.r),
        .g = static_cast<uint8_t>(ob_color.g),
        .b = static_cast<uint8_t>(ob_color.b)
    };
  };

  const auto index_sequence = std::views::iota(0, static_cast<int>(point_count));

  // par_unseq here is 3-5ms
  // seq is ~22ms

  std::for_each(std::execution::par_unseq, index_sequence.begin(), index_sequence.end(), transform_point);

//   tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, point_count - 1),
//                     [&](auto range) {
//                       for (int i = range.begin(); i < range.end(); i++) {
//                         std::cout << i << "\n";
//                       }
//                     });

// pc::logger()->error("Orbbec Kernel Not Implemented");

//   std::transform(std::execution::seq, indexed_points.begin(),
//                  indexed_points.end(), output_points.begin(), transform_point);

}

} // namespace pc::devices::orbbec