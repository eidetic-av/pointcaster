#pragma once

#include "orbbec_device_config.h"
#include <libobsensor/h/ObTypes.h>
#include <pointcaster/core_types.h>
#include <pointcaster/point_cloud.h>
#include <ranges>
#include <vector>

namespace pc::devices::orbbec {
using std::ranges::ref_view;
using std::ranges::zip_view;

// The input and output destination types the kernel expects to be zipped:

using IndexedFrameDataZipView =
    zip_view<std::span<const unsigned short>, std::span<const pc::color_rgb>,
             ref_view<const std::vector<int>>>;

using OutputZipView = zip_view<ref_view<std::vector<pc::position>>,
                               ref_view<std::vector<pc::color>>>;

// TODO
// the following constexpr instances of the zip_view type are just used to get
// the iterator type of the zipped inputs to use in function declarations for
// each kernel. seems like this is necessary because the iterator type of the
// zip_view type is marked private... maybe its a use case for 'friend'?
constexpr static std::vector<int> sequence_type_instance{};
constexpr static IndexedFrameDataZipView zip_view_type_instance{
    {}, {}, ref_view{sequence_type_instance}};

static std::vector<pc::position> positions_output_type_instance{};
static std::vector<pc::color> colors_output_type_instance{};
constexpr static OutputZipView output_view_type_instance{
    ref_view{positions_output_type_instance},
    ref_view{colors_output_type_instance}};

using IndexedFrameDataIterator = decltype(zip_view_type_instance.begin());
using OutputIterator = decltype(output_view_type_instance.begin());

// kernels

void input_transform(IndexedFrameDataIterator input_begin,
                     IndexedFrameDataIterator input_end,
                     OutputIterator output_begin);

void transform(const uint16_t *depth_frame_data_ptr,
               const color_rgb *color_frame_data_ptr, PointCloud &output_cloud,
               const OrbbecDeviceConfiguration &device_config,
               const OBCalibrationParam &calibration_parameters);

} // namespace pc::devices::orbbec