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

  // we dont need an extrinsic for the 2d->3d conversion because D2C has already
  // been performed on hardware
  constexpr static OBExtrinsic identity_extrinsic{
      {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
      {0.0f, 0.0f, 0.0f},
  };

  const auto frame_width =
      device_config.color_resolution ==
              OrbbecDeviceConfiguration::ColorResolution::FHD_1920x1080
          ? 1920
          : 1280;

  const auto indexed_points =
      std::ranges::zip_view(depth_data, color_data,
                            std::views::iota(0, static_cast<int>(point_count)));

  const auto output_points =
      std::ranges::zip_view(output_cloud.positions, output_cloud.colors);

  const auto transform_point = [&](auto point) {
    const auto [ob_depth, ob_color, i] = point;
    const auto x = i % frame_width;
    const auto y = i / frame_width;

    position p;
    OBPoint3f result;

    ob::CoordinateTransformHelper::transformation2dto3d(
        OBPoint2f(x, y), ob_depth, color_intrinsic, identity_extrinsic,
        &result);

    p.x = static_cast<int16_t>(result.x);
    p.y = -static_cast<int16_t>(result.y);
    p.z = -static_cast<int16_t>(result.z);

    color c;
    c.r = static_cast<uint8_t>(ob_color.r);
    c.g = static_cast<uint8_t>(ob_color.g);
    c.b = static_cast<uint8_t>(255);

    return std::make_tuple(p, c);
  };

  tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, point_count - 1),
                    [&](auto range) {
                      for (int i = range.begin(); i < range.end(); i++) {
                        std::cout << i << "\n";
                      }
                    });

pc::logger()->error("Orbbec Kernel Not Implemented");
//   std::transform(std::execution::par_unseq, indexed_points.begin(),
//                  indexed_points.end(), output_points.begin(), transform_point);

}

} // namespace pc::devices::orbbec