#pragma once

#include <array>
#include <serdepp/serde.hpp>
#include "structs.h"

using pc::types::int2;

enum class ScaleMode {
  Span, Letterbox,
  Count
};

enum class LetterboxMode {
  Aspect16x9, Aspect16x10,
  Count
};

struct PointCloudRendererConfiguration {
  bool unfolded = false;
  int2 resolution{3840, 2160};
  ScaleMode scale_mode = ScaleMode::Span;
  LetterboxMode letterbox_mode = LetterboxMode::Aspect16x9;
  float point_size = 0.0015f;
  bool ground_grid = true;
  bool skeletons = true;
  bool snapshots = true;

  DERIVE_SERDE(PointCloudRendererConfiguration,
	       (&Self::unfolded, "unfolded")
	       (&Self::resolution, "resolution")
	       (&Self::scale_mode, "scale_mode")
	       (&Self::letterbox_mode, "letterbox_mode")
	       (&Self::point_size, "point_size")
	       (&Self::ground_grid, "ground_grid")
	       (&Self::skeletons, "skeletons")
	       (&Self::snapshots, "snapshots"))

};
