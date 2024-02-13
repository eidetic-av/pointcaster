// https://github.com/catppuccin/catppuccin

#pragma once

#include <imgui.h>

constexpr ImVec4 rgb(int r, int g, int b, int a = 255) {
  return {r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f};
};

constexpr ImVec4 rgba(const auto &rgb, float a) {
  return {rgb.x, rgb.y, rgb.z, a};
};

namespace catpuccin {

constexpr auto mocha_text = rgb(205, 214, 244);
constexpr auto mocha_subtext = rgb(166, 173, 200);

constexpr auto mocha_crust = rgb(17, 17, 27);
constexpr auto mocha_mantle = rgb(24, 24, 37);
constexpr auto mocha_base = rgb(30, 30, 46);

constexpr auto mocha_surface = rgb(49, 50, 68);
constexpr auto mocha_surface1 = rgb(69, 71, 90);
constexpr auto mocha_surface2 = rgb(88, 91, 112);

constexpr auto mocha_overlay = rgb(108, 112, 134);

constexpr auto mocha_blue = rgb(137, 180, 250);
constexpr auto mocha_rosewater = rgb(245, 224, 220);
constexpr auto mocha_maroon = rgb(235, 160, 172);
constexpr auto mocha_peach = rgb(250, 179, 135);
constexpr auto mocha_lavender = rgb(180, 190, 254);
constexpr auto mocha_red = rgb(243, 139, 168);
constexpr auto mocha_yellow = rgb(249, 226, 175);

constexpr auto macchiato_base = rgb(36, 39, 58);
constexpr auto macchiato_mantle = rgb(30, 32, 48);
constexpr auto macchiato_crust = rgb(24, 25, 38);
constexpr auto macchiato_surface = rgb(54, 58, 79);
constexpr auto macchiato_surface1 = rgb(73, 77, 100);
constexpr auto macchiato_surface2 = rgb(91, 96, 120);

constexpr auto frappe_crust = rgb(35, 38, 52);
constexpr auto frappe_blue = rgb(140, 170, 238);

} // namespace catpuccin
