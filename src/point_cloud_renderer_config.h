#pragma once

#include <array>
#include <nlohmann/json.hpp>

enum class ScaleMode {
  Span, Letterbox,
  Count
};

enum class LetterboxMode {
  Aspect16x9, Aspect16x10,
  Count
};

struct PointCloudRendererConfiguration {
  std::array<int, 2> resolution{3840, 2160};
  ScaleMode scale_mode = ScaleMode::Span;
  LetterboxMode letterbox_mode = LetterboxMode::Aspect16x9;
  float point_size = 0.0015f;
  bool ground_grid = true;
  bool snapshots = true;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PointCloudRendererConfiguration, resolution,
                                   scale_mode, letterbox_mode, point_size,
                                   ground_grid, snapshots);
