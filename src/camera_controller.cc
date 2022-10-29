#include "camera_controller.h"
#include <numbers>
#include <spdlog/spdlog.h>
#include <Magnum/Math/Quaternion.h>

using namespace Magnum;

auto makeProjectionMatrix(Magnum::Deg fov) {
  return Matrix4::perspectiveProjection(fov, 4.0f / 3.0f, 0.01f, 200.0f);
}

CameraController::CameraController(Object3D &object) : Object3D{&object} {
  _yawObject = new Object3D{this};
  _pitchObject = new Object3D{_yawObject};

  // set initial orientation
  _yawObject->rotate(Rad(-std::numbers::pi / 2), Vector3::yAxis(1));
  _pitchObject->rotate(Rad(-std::numbers::pi / 6), Vector3::zAxis(1));

  _cameraObject = new Object3D{_pitchObject};
  _cameraObject->setTransformation(
      Matrix4::lookAt({7, 7, 0}, Vector3{}, Vector3::yAxis(1)));

  _camera = std::make_unique<SceneGraph::Camera3D>(*_cameraObject);
  _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend);
  _camera->setProjectionMatrix(makeProjectionMatrix(fov));

  translate(translation);
}

CameraController &CameraController::rotate(const Magnum::Vector2i &shift) {
  Vector2 s = Vector2{shift} * _speed;
  _yawObject->rotate(Rad(s.x()), Vector3::yAxis(1));
  _pitchObject->rotate(Rad(s.y()), Vector3::zAxis(1));
  return *this;
}

CameraController &CameraController::move(const Magnum::Vector2i &shift) {
  Vector2 s = Vector2{shift} * _speed;
  // TODO this is all such a mess
  auto yaw = Quaternion::fromMatrix(_yawObject->transformationMatrix().rotation())
	  .toEuler().y();
  auto yaw_d = (float) Deg(yaw);
  auto sum = (_yawObject->transformationMatrix().rotation()).toVector().sum();
  // spdlog::info("yaw: {}, sum: {}", (float) Deg(yaw), sum);
  auto sign_x = (sum > 1 ? -1 : 1);
  auto flip_x = (yaw_d < 0 && sum > 1) ? -1 : 1;
  auto flip_z = (yaw_d > -90 && yaw_d < 0 && sum > 1) ? -1 : 1;
  // spdlog::info("flip_x: {}, flip_z: {}", flip_x, flip_z);
  auto z_scale_x = std::fmod(std::abs((yaw_d + 90) / 90 * sign_x * flip_x), 1.0f);
  auto x_scale_x = std::fmod(std::abs(1 - (z_scale_x * sign_x * flip_x)), 1.0f);
  // spdlog::info("z_scale_x: {}, x_scale_x: {}", z_scale_x, x_scale_x);
  auto xRot = (float)Deg(
      Quaternion::fromMatrix(_pitchObject->transformationMatrix().rotation())
          .toEuler().z());
  translate({s.x() * x_scale_x, s.y(), s.x() * z_scale_x * flip_z});
  return *this;
}

CameraController &CameraController::zoom(const Magnum::Float &delta) {
  fov += Deg(-delta);
  _camera->setProjectionMatrix(makeProjectionMatrix(fov));
  return *this;
}
