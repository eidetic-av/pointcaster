#pragma once

#include <string>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Vector3.h>

namespace pc::camera {

using Euler = Magnum::Math::Vector3<Magnum::Math::Rad<float>>;
using Position = Magnum::Math::Vector3<float>;
using Deg_f = Magnum::Math::Deg<float>;
using Rad_f = Magnum::Math::Rad<float>;

namespace defaults {

static constexpr Euler rotation{Deg_f{15}, Deg_f{0}, Deg_f{0}};
static constexpr float distance = 10.0f;
static const Position translation{
    0.0f, distance *std::sin(float(Rad_f(rotation.x()))),
    distance *std::cos(float(Rad_f(rotation.x())))};

static constexpr Deg_f fov{45};

} // namespace defaults

struct CameraConfiguration {
  std::string id;
  std::string name;
  Euler rotation = defaults::rotation;
  Position translation = defaults::translation;
  Deg_f fov = defaults::fov;
};

} // namespace pc::camera
