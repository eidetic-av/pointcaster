#pragma once

#include <functional>
#include <pointcaster/core_types.h>
#include <span>
#include <tuple>

namespace pc {

enum class BackendType { CPU, CUDA };

namespace backend {

struct CameraIntrinsics {
  float fx, fy, cx, cy;
  float inv_fx, inv_fy;
  size_t frame_width;
};

struct Extrinsics {
  float r[9]; // row-major 3x3
  float t[3];
};

using PointType = std::tuple<position, color>;

// return an internal point type given pixel data from a depth frame
using PointTransformer =
    std::function<PointType(const int, const uint16_t, const color_rgb &)>;

} // namespace backend

} // namespace pc
