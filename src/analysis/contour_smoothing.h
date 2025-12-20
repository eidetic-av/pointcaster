#pragma once

#include "analyser_2d_config.gen.h"
#include <opencv2/core/types.hpp>
#include <vector>


namespace pc::analysis::contour_smoothing {

// Chaikin's algorithm for iterative contour smoothing
std::vector<cv::Point2f>
chaikin_smooth(const std::vector<cv::Point2f> &contour_input,
               const ContourSmoothingConfiguration &config);

// resample a 2d contour to have a specific number of output vertices
// (arc length reparametisation)
std::vector<cv::Point2f>
resample_by_arclength(const std::vector<cv::Point2f> &contour_input,
                      int resample_count);

} // namespace pc::analysis::contour_smoothing
