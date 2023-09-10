#pragma once

#include "../structs.h"

#include <serdepp/serde.hpp>

#include <k4a/k4atypes.h>

namespace pc::sensors {

using pc::types::float3;
using pc::types::minMax;
using pc::types::short3;

struct BodyTrackingConfiguration {
  bool unfolded = false;
  bool enabled;

  DERIVE_SERDE(BodyTrackingConfiguration, (&Self::unfolded, "unfolded")
	       (&Self::enabled, "enabled"))
};

struct K4AConfiguration {
  bool unfolded = false;
  k4a_depth_mode_t depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
  bool auto_tilt = false;

  DERIVE_SERDE(K4AConfiguration, (&Self::unfolded, "unfolded")
	       (&Self::depth_mode, "depth_mode")
	       (&Self::auto_tilt, "auto_tilt"))
};

struct DeviceConfiguration {
  bool unfolded = false;

  bool flip_x = false;
  bool flip_y = false;
  bool flip_z = false;
  minMax<short> crop_x{-10000, 10000};
  minMax<short> crop_y{-10000, 10000};
  minMax<short> crop_z{-10000, 10000};
  float3 offset{0, 0, 0};
  float3 rotation_deg{0, 0, 0};
  float scale = 1;
  int sample = 1;

  BodyTrackingConfiguration body;
  K4AConfiguration k4a;

  DERIVE_SERDE(DeviceConfiguration,
	       (&Self::unfolded, "unfolded")
	       (&Self::flip_x, "flip_x") (&Self::flip_y, "flip_y") (&Self::flip_z, "flip_z")
	       (&Self::crop_x, "crop_x") (&Self::crop_y, "crop_y") (&Self::crop_z, "crop_z")
	       (&Self::offset, "offset") (&Self::rotation_deg, "rotation_deg") (&Self::scale, "scale")
	       (&Self::sample, "sample") (&Self::body, "body") (&Self::k4a, "k4a"))
};

} // namespace pc::sensors
