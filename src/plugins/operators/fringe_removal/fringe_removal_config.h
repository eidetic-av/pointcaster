#pragma once

#include <camera/look_at_camera_config.h>
#include <config/color_transform_config.h>
#include <plugins/backend/backend_types.h>
#include <rfl/Literal.hpp>
#include <string>

namespace pc::operators {

class FringeRemovalOperator;

struct FringeRemovalConfiguration {
  std::string id;                      // @hidden
  rfl::DefaultVal<std::string> label;  // @hidden
  rfl::DefaultVal<bool> active = true; // @hidden

  LookAtCameraConfiguration camera;

  // edge detection
  rfl::DefaultVal<float> depth_threshold = 0.02f; // @minmax(0.005, 0.2)
  rfl::DefaultVal<int> max_search_neighbors = 50; // @minmax(5, 200)
  rfl::DefaultVal<float> canny_low = 40.0f;       // @minmax(1, 255)
  rfl::DefaultVal<float> canny_high = 100.0f;     // @minmax(1, 255)

  // fringe removal
  rfl::DefaultVal<bool> remove_occluding_fringe = true;
  rfl::DefaultVal<int> occluding_erosion_px = 1; // @minmax(0, 20)
  rfl::DefaultVal<bool> remove_canny_fringe = false;
  rfl::DefaultVal<int> canny_proximity_px = 2; // @minmax(0, 20)
  rfl::DefaultVal<int> canny_erosion_px = 1;   // @minmax(0, 20)

  // mask smoothing
  rfl::DefaultVal<int> morph_close_radius = 2;    // @minmax(0, 10)
  rfl::DefaultVal<int> blur_radius = 3;           // @minmax(0, 10)
  rfl::DefaultVal<float> blur_threshold = 128.0f; // @minmax(1, 254)

  rfl::DefaultVal<float> removal_depth_range = 0.0f; // @minmax(0, 10)

  rfl::DefaultVal<BackendType> backend = BackendType::CPU;

  using OperatorType = FringeRemovalOperator;
  using Tag = rfl::Literal<"fringeRemoval">;
  static constexpr auto PluginName = "FringeRemovalOperator";
};

} // namespace pc::operators