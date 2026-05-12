#pragma once

#include <functional>
#include <optional>
#include <pointcaster/core_types.h>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>


namespace pc::camera {

struct CameraFrameData {
  int width;
  int height;

  std::unordered_map<std::string, std::vector<color>> color_buffers;
  std::unordered_map<std::string, std::vector<float>> float_buffers;
  std::unordered_map<std::string, std::vector<int32_t>> int_buffers;
};

struct CameraFrame {
  std::string name;
  std::optional<CameraFrameData> frame_data;
};

using CameraFrameRef = std::reference_wrapper<CameraFrame>;

} // namespace pc::camera