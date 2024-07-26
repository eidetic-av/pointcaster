// https://github.com/catppuccin/catppuccin

#pragma once

#include <imgui.h>
#include <Magnum/Math/Color.h>

namespace catpuccin {

template <typename VectorType>
constexpr VectorType rgb(int r, int g, int b, int a = 255) {
  return VectorType{r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f};
};

template <typename VectorType>
constexpr VectorType rgba(const auto &rgb, float a) {
  return {rgb.x, rgb.y, rgb.z, a};
};

constexpr uint32_t to_uint(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return (r << 24) | (g << 16) | (b << 8) | a;
}

namespace imgui {

	constexpr auto mocha_text = rgb<ImVec4>(205, 214, 244);
	constexpr auto mocha_subtext = rgb<ImVec4>(166, 173, 200);

	constexpr auto mocha_crust = rgb<ImVec4>(17, 17, 27);
	constexpr auto mocha_mantle = rgb<ImVec4>(24, 24, 37);
	constexpr auto mocha_base = rgb<ImVec4>(30, 30, 46);

	constexpr auto mocha_surface = rgb<ImVec4>(49, 50, 68);
	constexpr auto mocha_surface1 = rgb<ImVec4>(69, 71, 90);
	constexpr auto mocha_surface2 = rgb<ImVec4>(88, 91, 112);

	constexpr auto mocha_overlay = rgb<ImVec4>(108, 112, 134);

	constexpr auto mocha_blue = rgb<ImVec4>(137, 180, 250);
	constexpr auto mocha_rosewater = rgb<ImVec4>(245, 224, 220);
	constexpr auto mocha_maroon = rgb<ImVec4>(235, 160, 172);
	constexpr auto mocha_peach = rgb<ImVec4>(250, 179, 135);
	constexpr auto mocha_lavender = rgb<ImVec4>(180, 190, 254);
	constexpr auto mocha_red = rgb<ImVec4>(243, 139, 168);
	constexpr auto mocha_yellow = rgb<ImVec4>(249, 226, 175);

	constexpr auto macchiato_base = rgb<ImVec4>(36, 39, 58);
	constexpr auto macchiato_mantle = rgb<ImVec4>(30, 32, 48);
	constexpr auto macchiato_crust = rgb<ImVec4>(24, 25, 38);
	constexpr auto macchiato_surface = rgb<ImVec4>(54, 58, 79);
	constexpr auto macchiato_surface1 = rgb<ImVec4>(73, 77, 100);
	constexpr auto macchiato_surface2 = rgb<ImVec4>(91, 96, 120);

	constexpr auto frappe_crust = rgb<ImVec4>(35, 38, 52);
	constexpr auto frappe_blue = rgb<ImVec4>(140, 170, 238);

} // namespace imgui

namespace imnodes {
	constexpr auto frappe_crust = to_uint(35, 38, 52);
	constexpr auto frappe_blue = to_uint(140, 170, 238);
}

namespace magnum {

using Color4f = Magnum::Math::Color4<float>;

constexpr auto mocha_text = rgb<Color4f>(205, 214, 244);
constexpr auto mocha_subtext = rgb<Color4f>(166, 173, 200);

constexpr auto mocha_crust = rgb<Color4f>(17, 17, 27);
constexpr auto mocha_mantle = rgb<Color4f>(24, 24, 37);
constexpr auto mocha_base = rgb<Color4f>(30, 30, 46);

constexpr auto mocha_surface = rgb<Color4f>(49, 50, 68);
constexpr auto mocha_surface1 = rgb<Color4f>(69, 71, 90);
constexpr auto mocha_surface2 = rgb<Color4f>(88, 91, 112);

constexpr auto mocha_overlay = rgb<Color4f>(108, 112, 134);

constexpr auto mocha_blue = rgb<Color4f>(137, 180, 250);
constexpr auto mocha_rosewater = rgb<Color4f>(245, 224, 220);
constexpr auto mocha_maroon = rgb<Color4f>(235, 160, 172);
constexpr auto mocha_peach = rgb<Color4f>(250, 179, 135);
constexpr auto mocha_lavender = rgb<Color4f>(180, 190, 254);
constexpr auto mocha_red = rgb<Color4f>(243, 139, 168);
constexpr auto mocha_yellow = rgb<Color4f>(249, 226, 175);

constexpr auto macchiato_base = rgb<Color4f>(36, 39, 58);
constexpr auto macchiato_mantle = rgb<Color4f>(30, 32, 48);
constexpr auto macchiato_crust = rgb<Color4f>(24, 25, 38);
constexpr auto macchiato_surface = rgb<Color4f>(54, 58, 79);
constexpr auto macchiato_surface1 = rgb<Color4f>(73, 77, 100);
constexpr auto macchiato_surface2 = rgb<Color4f>(91, 96, 120);

constexpr auto frappe_crust = rgb<Color4f>(35, 38, 52);
constexpr auto frappe_blue = rgb<Color4f>(140, 170, 238);

} // namespace magnum

} // namespace catpuccin
