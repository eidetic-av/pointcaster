#pragma once

#include "../serialization.h"
#include "../structs.h"
#include <array>

using pc::types::Float2;
using pc::types::Int2;

enum class ScaleMode { Span = 0, Letterbox = 1, Count = 2 };

enum class LetterboxMode { Aspect16x9 = 0, Aspect16x10 = 1, Count = 2 };

using MinMaxFloat = pc::types::MinMax<float>;

struct PointCloudRendererConfiguration {
  bool unfolded{false};
  Int2 resolution{3840, 2160}; // @minmax(2, 3840)
  int scale_mode{(int)ScaleMode::Span}; // @minmax(0, 1)
  int letterbox_mode{(int)LetterboxMode::Aspect16x9}; // @minmax(0, 1)
  bool orthographic{false};
  Float2 orthographic_size{5, 5};
  MinMaxFloat clipping{0.001f, 200.0f};
  float point_size{0.0015f}; // @minmax(0.0001f, 0.05f)
  bool ground_grid{true};
  bool skeletons{true}; // @hidden
  bool snapshots{true}; // @hidden
};
