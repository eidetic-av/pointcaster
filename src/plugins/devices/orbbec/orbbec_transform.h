// #pragma once

// #include "orbbec_device_config.h"
// #include <functional>
// #include <libobsensor/h/ObTypes.h>
// #include <pointcaster/core_types.h>
// #include <pointcaster/point_cloud.h>
// #include <span>
// #include <vector>

// namespace pc::devices::orbbec {

// using UShortDepthData = std::span<const uint16_t>;
// using RGBColorData = std::span<const color_rgb>;

// using PointType = std::tuple<position, color>;
// using PointGenerator = std::function<PointType(const int)>;

// void transform_cpu(PointCloud &output_cloud,
//                    const PointGenerator &generate_point);

// void transform_cpu(std::span<const uint16_t> ob_depth_data,
//                    std::span<const color_rgb> ob_color_data,
//                    PointCloud &output_cloud,
//                    const OrbbecDeviceConfiguration &device_config,
//                    const OBCalibrationParam &calibration_parameters);

// void transform_gpu(std::span<const uint16_t> ob_depth_data,
//                    std::span<const color_rgb> ob_color_data,
//                    PointCloud &output_cloud,
//                    const OrbbecDeviceConfiguration &device_config,
//                    const OBCalibrationParam &calibration_parameters);

// } // namespace pc::devices::orbbec