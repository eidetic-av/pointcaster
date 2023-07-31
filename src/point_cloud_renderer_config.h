#pragma once

#include <array>

struct PointCloudRendererConfiguration {
  std::array<int, 2> resolution;
  float point_size = 0.0015f;
  bool ground_grid = true;
};
