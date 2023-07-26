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
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Vector3.h>
#include <atomic>
#include <memory>
#include "camera_config.h"
#include <Magnum/Math/Vector3.h>

namespace pc::camera {

typedef Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation3D>
    Object3D;
typedef Magnum::SceneGraph::Scene<Magnum::SceneGraph::MatrixTransformation3D>
    Scene3D;

class CameraController : public Object3D {
public:
  static std::atomic<uint> count;

  CameraController(Magnum::Platform::Application *app, Object3D &object);
  CameraController(Magnum::Platform::Application *app,
                            Object3D &object, CameraConfiguration config);

  ~CameraController();

  const CameraConfiguration &config() const { return _config; };
  const std::string name() const { return _config.name; };
  Magnum::SceneGraph::Camera3D &camera() const { return *_camera; }

  void setRotation(const Magnum::Math::Vector3<Magnum::Math::Rad<float>>& rotation);
  void setTranslation(const Magnum::Math::Vector3<float>& translation);
  void dolly(Magnum::Platform::Sdl2Application::MouseScrollEvent &event);

  void mouseRotate(Magnum::Platform::Sdl2Application::MouseMoveEvent &event);
  void mouseTranslate(Magnum::Platform::Sdl2Application::MouseMoveEvent &event);

  // 0 is regular perspective, 1 is orthographic
  CameraController &setPerspective(const Magnum::Float &value);
  CameraController &zoomPerspective(Magnum::Platform::Sdl2Application::MouseScrollEvent &event);

  void setupFramebuffer(const Magnum::Vector2i frame_size);
  void bindFramebuffer();

  Magnum::GL::Texture2D& outputFrame();

private:
  CameraConfiguration _config;

  Magnum::Platform::Application *_app;

  std::unique_ptr<Object3D> _camera_parent;
  std::unique_ptr<Magnum::SceneGraph::Camera3D> _camera;

  std::unique_ptr<Object3D> _yaw_parent;
  std::unique_ptr<Object3D> _pitch_parent;
  std::unique_ptr<Object3D> _roll_parent;

  Magnum::Vector2i _frame_size;
  std::unique_ptr<Magnum::GL::Texture2D> _color;
  std::unique_ptr<Magnum::GL::Renderbuffer> _depth_stencil;
  std::unique_ptr<Magnum::GL::Framebuffer> _framebuffer;

  Magnum::Vector2 _rotate_speed{-0.0035f, 0.0035f};
  Magnum::Vector2 _move_speed{-0.0035f, 0.0035f};

  Magnum::Deg _fov{45};
  Magnum::Float _perspective_value{0.5};
  Magnum::Vector3 _translation{};

  Magnum::Matrix4 make_projection_matrix();

  Magnum::Vector3 unproject(const Magnum::Vector2i &window_position,
                            Magnum::Float depth) const;
  Magnum::Float depth_at(const Magnum::Vector2i &window_position);
};

} // namespace pc::camera
