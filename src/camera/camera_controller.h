#pragma once

#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Scene.h>
#include <atomic>
#include <memory>

namespace pc::camera {

typedef Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>
    Object3D;
typedef Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>
    Scene3D;

class CameraController : public Object3D {
public:
  static std::atomic<uint> count;

  std::string name;

  explicit CameraController(Magnum::Platform::Application *app,
                            Object3D &object);

  ~CameraController();

  Magnum::SceneGraph::Camera3D &camera() const { return *_camera; }

  Object3D &cameraObject() const { return *_cameraObject; }

  CameraController &rotate(const Magnum::Vector2i &shift);
  CameraController &move(Magnum::Platform::Sdl2Application::MouseMoveEvent &event);
  CameraController &dolly(Magnum::Platform::Sdl2Application::MouseScrollEvent &event);

  // 0 is regular perspective, 1 is orthographic
  CameraController &setPerspective(const Magnum::Float &value);
  CameraController &zoomPerspective(Magnum::Platform::Sdl2Application::MouseScrollEvent &event);

  CameraController &setSpeed(const Magnum::Vector2 &speed);

  void setupFramebuffer(const Magnum::Vector2i frame_size);

  Magnum::Vector2i _frame_size;
  std::unique_ptr<Magnum::GL::Texture2D> _color;
  std::unique_ptr<Magnum::GL::Renderbuffer> _depth_stencil;
  std::unique_ptr<Magnum::GL::Framebuffer> _framebuffer;

private:
  Magnum::Platform::Application *_app;

  Object3D *_yawObject;
  Object3D *_pitchObject;
  Object3D *_cameraObject;

  std::unique_ptr<Magnum::SceneGraph::Camera3D> _camera;
  Magnum::Vector2 _rotate_speed{-0.0035f, 0.0035f};
  Magnum::Vector2 _move_speed{-0.0035f, 0.0035f};

  Magnum::Deg _fov{45};
  Magnum::Float _perspective_value{0.5};
  Magnum::Vector3 _translation{};

  Magnum::Matrix4 make_projection_matrix();

  Magnum::Vector3 _rotation_point;
  Magnum::Vector3 unproject(const Magnum::Vector2i &window_position,
                            Magnum::Float depth) const;
  Magnum::Float depth_at(const Magnum::Vector2i &window_position);
};

} // namespace pc::camera
