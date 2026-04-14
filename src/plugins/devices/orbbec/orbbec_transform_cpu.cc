// #include "orbbec_transform.h"
// #include "plugins/devices/orbbec/orbbec_device_config.h"

// #include <algorithm>
// #include <execution>
// #include <iostream>
// #include <libobsensor/h/ObTypes.h>
// #include <libobsensor/hpp/Utils.hpp>
// #include <logger/logger.h>
// #include <mutex>
// #include <ranges>
// #include <span>
// #include <thread>
// #include <oneapi/tbb/parallel_for.h>

// // #include <thrust/copy.h>
// // #include <thrust/device_vector.h>
// // #include <thrust/fill.h>
// // #include <thrust/functional.h>
// // #include <thrust/replace.h>
// // #include <thrust/sequence.h>
// // #include <thrust/transform.h>

// #include <oneapi/tbb/blocked_range.h>
// #include <oneapi/tbb/parallel_for.h>

// namespace pc::devices::orbbec {

// void transform_cpu(PointCloud &output_cloud,
//                    const PointGenerator &generate_point) {

//   const auto point_count = output_cloud.size();

//   const auto index_sequence =
//       std::views::iota(0, static_cast<int>(point_count));

//   const auto transform_point = [&](const auto index) {
//     auto [pos, col] = generate_point(index);
//     // TODO
//     // do transform stuff here
//     output_cloud.positions[index] = std::move(pos);
//     output_cloud.colors[index] = std::move(col);
//   };

//   std::for_each(std::execution::par_unseq, index_sequence.begin(),
//                 index_sequence.end(), transform_point);
// };

// void transform_cpu(std::span<const uint16_t> ob_depth_data,
//                    std::span<const color_rgb> ob_color_data,
//                    PointCloud &output_cloud,
//                    const OrbbecDeviceConfiguration &device_config,
//                    const OBCalibrationParam &calibration_parameters) {

//   const auto point_count = output_cloud.size();

//   const auto color_intrinsic =
//       calibration_parameters.intrinsics[OB_SENSOR_COLOR];

//   // we dont need an extrinsic for the 2d->3d conversion because frame alignment
//   // has already been performed within orbbec sdk
//   constexpr static OBExtrinsic identity_extrinsic{
//       {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
//       {0.0f, 0.0f, 0.0f},
//   };

//   // TODO hardcoded with depth -> color, so color res is used everywhere
//   const auto frame_width =
//       std::get<0>(orbbec::resolution(device_config.color_resolution));

//   const auto transform_point = [&](const auto i) {
//     const auto x = i % frame_width;
//     const auto y = i / frame_width;

//     const auto ob_depth = ob_depth_data[i];
//     const auto ob_color = ob_color_data[i];

//     OBPoint3f result;

//     ob::CoordinateTransformHelper::transformation2dto3d(
//         OBPoint2f(x, y), ob_depth, color_intrinsic, identity_extrinsic,
//         &result);

//     output_cloud.positions[i] = {.x = static_cast<int16_t>(result.x),
//                                  .y = -static_cast<int16_t>(result.y),
//                                  .z = -static_cast<int16_t>(result.z)};

//     output_cloud.colors[i] = {.r = static_cast<uint8_t>(ob_color.r),
//                               .g = static_cast<uint8_t>(ob_color.g),
//                               .b = static_cast<uint8_t>(ob_color.b)};
//   };

//   const auto index_sequence =
//       std::views::iota(0, static_cast<int>(point_count));

//   std::for_each(std::execution::par_unseq, index_sequence.begin(),
//                 index_sequence.end(), transform_point);
// }

// } // namespace pc::devices::orbbec