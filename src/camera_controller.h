#pragma once

#include <Magnum/Math/Angle.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>
#include <memory>

typedef Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>
    Object3D;
typedef Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>
    Scene3D;

class CameraController : public Object3D {
public:
  explicit CameraController(Object3D &object);

  Magnum::SceneGraph::Camera3D &camera() const { return *_camera; }

  Object3D &cameraObject() const { return *_cameraObject; }

  CameraController &rotate(const Magnum::Vector2i &shift);
  CameraController &move(const Magnum::Vector2i &shift);
  CameraController &zoom(const Magnum::Float &delta);

  CameraController &setSpeed(const Magnum::Vector2 &speed) {
    _speed = speed;
    return *this;
  }

private:
  Object3D *_yawObject;
  Object3D *_pitchObject;
  Object3D *_cameraObject;

  std::unique_ptr<Magnum::SceneGraph::Camera3D> _camera;
  Magnum::Vector2 _speed{-0.0035f, 0.0035f};

  constexpr static auto start_fov = Magnum::Deg(45.0f);
  Magnum::Deg fov = start_fov;

  Magnum::Vector3 translation{0, 1, 0};
};
