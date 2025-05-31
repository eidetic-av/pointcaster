#pragma once

#include "../structs.h"
#include <array>
#include <thrust/tuple.h>

namespace pc::filters {

using pc::types::color;
using pc::types::position;
using indexed_point_t = thrust::tuple<position, color, int>;

struct chroma_key {
  float target_hue_degrees;
  float hue_width_degrees;
  float minimum_saturation;
  float minimum_value;
  bool invert_mask;

  struct hsv {
    float h, s, v;
  };

  __device__ static hsv rgb_to_hsv(const color &rgba) {
    auto [red, green, blue, alpha] = rgba;
    const float fr = red / 255.f;
    const float fg = green / 255.f;
    const float fb = blue / 255.f;

    const float max_c = fmaxf(fr, fmaxf(fg, fb));
    const float min_c = fminf(fr, fminf(fg, fb));
    const float delta = max_c - min_c;

    const float value = max_c;
    const float saturation = (max_c > 0.f ? delta / max_c : 0.f);

    float hue;
    if (delta < 1e-5f) {
      hue = 0.f;
    } else if (max_c == fr) {
      hue = 60.f * fmodf(((fg - fb) / delta), 6.f);
    } else if (max_c == fg) {
      hue = 60.f * (((fb - fr) / delta) + 2.f);
    } else {
      hue = 60.f * (((fr - fg) / delta) + 4.f);
    }
    if (hue < 0.f) hue += 360.f;

    return {hue, saturation, value};
  }

  __device__ bool operator()(const indexed_point_t &point) const {
    const color pixel_color = thrust::get<1>(point);
    auto input_color = rgb_to_hsv(pixel_color);
    auto hue = input_color.h;
    auto saturation = input_color.s;
    auto value = input_color.v;

    // shortest difference around the circle
    float delta_hue = fabsf(hue - target_hue_degrees);
    if (delta_hue > 180.f) delta_hue = 360.f - delta_hue;

    const bool inside_hue_band = (delta_hue <= hue_width_degrees * 0.5f);
    const bool above_thresholds =
        (saturation >= minimum_saturation && value >= minimum_value);

    bool keep_pixel = !(inside_hue_band && above_thresholds);
    if (invert_mask) { keep_pixel = !keep_pixel; }
    return keep_pixel;
  }
};

} // namespace pc::filters
