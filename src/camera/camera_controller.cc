#include "camera_controller.h"
#include "../log.h"
#include "../math.h"
#include "../uuid.h"
#include "camera_config.h"
#include <algorithm>
#include <numbers>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/Image.h>
#include <Magnum/Math/FunctionsBatch.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Math/Vector3.h>

namespace pc::camera {

using namespace Magnum;
using Magnum::Matrix4;
using Magnum::Quaternion;
using Magnum::Vector3;
using Magnum::Math::Deg;

std::atomic<uint> CameraController::count = 0;

CameraController::CameraController(Magnum::Platform::Application *app,
				   Object3D &object)
    : CameraController(app, object, CameraConfiguration{}){};

CameraController::CameraController(Magnum::Platform::Application *app,
				   Object3D &object, CameraConfiguration config)
    : _app(app), _config(config), Object3D{&object} {

  // rotations are manipulated by individual parent objects...
  // this makes rotations easier to reason about and serialize,
  // as we don't need quaternion multiplications or conversions to and from
  // matrices to serialize as three independent euler values

  _yaw_parent = std::make_unique<Object3D>(this);
  _pitch_parent = std::make_unique<Object3D>(_yaw_parent.get());
  _camera_parent = std::make_unique<Object3D>(_pitch_parent.get());
  _roll_parent = std::make_unique<Object3D>(_camera_parent.get());
  _camera = std::make_unique<SceneGraph::Camera3D>(*_roll_parent);

  _camera_parent->setTransformation(Matrix4::lookAt(
      {defaults::magnum::translation.z(), defaults::magnum::translation.y(),
       defaults::magnum::translation.x()},
      {}, Vector3::yAxis()));

  _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend);

  // deserialize our camera configuration into Magnum types
  auto rotation = Euler{Deg_f(_config.rotation[0]), Deg_f(_config.rotation[1]),
			Deg_f(_config.rotation[2])};
  auto translation = Position{_config.translation[0], _config.translation[1],
		      _config.translation[2]};
  auto fov = Deg_f(_config.fov);

  setRotation(rotation);
  setTranslation(translation);

  _camera->setProjectionMatrix(
      Matrix4::perspectiveProjection(fov, 4.0f / 3.0f, 0.001f, 200.0f));

  if (_config.id.empty()) {
    _config.id = pc::uuid::word();
  }
  _config.name = "camera_" + std::to_string(++CameraController::count);

  pc::log.info("Initialised Camera Controller '%s' with id '%s'", _config.name,
               _config.id);
}

CameraController::~CameraController() { CameraController::count--; }

void CameraController::setupFramebuffer(Vector2i frame_size) {
  if (frame_size == _frame_size)
    return;
  _frame_size = frame_size;

  _color = std::make_unique<GL::Texture2D>();
  _color->setStorage(1, GL::TextureFormat::RGBA8, frame_size);

  _depth_stencil = std::make_unique<GL::Renderbuffer>();
  _depth_stencil->setStorage(GL::RenderbufferFormat::Depth24Stencil8,
                             frame_size);
  _framebuffer = std::make_unique<GL::Framebuffer>(Range2Di{{}, frame_size});
  _framebuffer->attachTexture(GL::Framebuffer::ColorAttachment{0}, *_color, 0);
  _framebuffer->attachRenderbuffer(
      GL::Framebuffer::BufferAttachment::DepthStencil, *_depth_stencil);
}

void CameraController::bindFramebuffer() {
  _framebuffer->clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth)
      .bind();
}

GL::Texture2D &CameraController::outputFrame() { return *_color; }

// TODO
bool _is_dz_started = false;
float _init_height_at_distance = 0.0f;

Matrix4 CameraController::make_projection_matrix() {
  auto frustum_height_at_distance = [](float distance, float fov) {
    return 2.0f * distance *
           std::tan(fov * 0.5f * Math::Constants<float>::pi() / 180.0f);
  };

  auto fov_for_height_and_distance = [](float height, float distance) {
    return 2.0f * std::atan(height * 0.5f / distance) * 180.0f /
           Math::Constants<float>::pi();
  };

  const auto focal_point =
      unproject(_frame_size / 2, depth_at(_frame_size / 2));
  auto camera_location = _camera_parent->transformation().translation();
  pc::log.info("camera_location: %f, %f, %f", camera_location.x(),
               camera_location.y(), camera_location.z());
  const auto target_distance = (camera_location - focal_point).length();

  if (!_is_dz_started) {
    _init_height_at_distance =
        frustum_height_at_distance(target_distance, _perspective_value);
    _is_dz_started = true;
  }

  auto fov = pc::math::remap(0.0f, 1.0f, 0.01f, 90.0f - 0.01f,
                             _perspective_value, true);

  auto height = frustum_height_at_distance(target_distance, fov);
  auto new_fov = fov_for_height_and_distance(height, target_distance);

  // Compute new distance to maintain the initial frustum height.
  // auto new_distance = _init_height_at_distance / (2.0f *
  // Math::tan(Deg(new_fov * 0.5f) * Math::Constants<float>::pi() / 180.0f));
  // Compute the new camera position to move towards or away from the subject as
  // FOV changes.
  auto direction = (camera_location - focal_point).normalized();
  // camera_location = focal_point + direction * -new_distance;

  // Update the camera's position.
  // auto transform = Matrix4::from(_camera_parent->transformation().rotation(),
  // 				 camera_location);
  // _camera_parent->setTransformation(transform);

  // return Matrix4::perspectiveProjection(Deg(new_fov), 4.0f / 3.0f, 0.001f,
  return Matrix4::perspectiveProjection(
      Deg(new_fov), _frame_size.x() / _frame_size.y(), 0.001f, 200.0f);
}

void CameraController::setRotation(
    const Magnum::Math::Vector3<Magnum::Math::Rad<float>> &rotation) {

  _config.rotation = {float(Deg_f(rotation.x())), float(Deg_f(rotation.y())),
		      float(Deg_f(rotation.z()))};

  _yaw_parent->resetTransformation();
  auto y_rotation = rotation.y() - defaults::magnum::rotation.y();
  _yaw_parent->rotate(y_rotation, Vector3::yAxis());

  _pitch_parent->resetTransformation();
  auto x_rotation = rotation.x() - defaults::magnum::rotation.x();
  _pitch_parent->rotate(x_rotation, Vector3::zAxis());

  _roll_parent->resetTransformation();
  auto z_rotation = rotation.z() - defaults::magnum::rotation.z();
  _roll_parent->rotate(z_rotation, Vector3::zAxis());
}

void CameraController::setTranslation(
    const Magnum::Math::Vector3<float> &translation) {
  _config.translation = {translation.x(), translation.y(), translation.z()};
  auto transform = _camera_parent->transformationMatrix();
  transform.translation() = {translation.z(), translation.y(), translation.x()};
  _camera_parent->setTransformation(transform);
}

void CameraController::dolly(
    Magnum::Platform::Sdl2Application::MouseScrollEvent &event) {
  const auto delta = event.offset().y();
  const auto frame_centre = _frame_size / 2;
  const auto centre_depth = depth_at(frame_centre);
  const auto focal_point = unproject(frame_centre, centre_depth);
  constexpr auto speed = 0.01f;
  _camera_parent->translateLocal(focal_point * delta * speed);
  auto translation = _camera_parent->transformationMatrix().translation();
  _config.translation = {translation.z(), translation.y(), translation.x()};
}

void CameraController::mouseRotate(
    Magnum::Platform::Sdl2Application::MouseMoveEvent &event) {
  auto delta = Vector2{event.relativePosition()} * _rotate_speed;
  Euler rotation_amount;
  if (event.modifiers() ==
      Magnum::Platform::Sdl2Application::InputEvent::Modifier::Ctrl) {
    rotation_amount.z() = Rad(delta.y());
  } else {
    rotation_amount.x() = Rad(-delta.y());
    rotation_amount.y() = Rad(delta.x());
  }
  auto rotation = Euler{Deg_f(_config.rotation[0]), Deg_f(_config.rotation[1]),
			Deg_f(_config.rotation[2])};
  setRotation(rotation + rotation_amount);
}

void CameraController::mouseTranslate(
    Magnum::Platform::Sdl2Application::MouseMoveEvent &event) {
  const auto frame_centre = _frame_size / 2;
  const auto centre_depth = depth_at(frame_centre);
  const Vector3 p = unproject(event.position(), centre_depth);
  const auto delta =
      Vector3{(float)event.relativePosition().x() * _move_speed.x(),
	      (float)event.relativePosition().y() * _move_speed.y(), 0};
  auto translation = Position{_config.translation[0], _config.translation[1],
		      _config.translation[2]};
  setTranslation(translation + delta);
}

CameraController &
CameraController::setPerspective(const Magnum::Float &perspective_value) {
  _perspective_value = Math::max(Math::min(perspective_value, 1.0f),
                                 std::numeric_limits<float>::min());
  _camera->setProjectionMatrix(make_projection_matrix());
  return *this;
}

CameraController &CameraController::zoomPerspective(
    Magnum::Platform::Sdl2Application::MouseScrollEvent &event) {
  auto delta = event.offset().y();
  setPerspective(_perspective_value - delta / 10);
  return *this;
}

Magnum::Vector3
CameraController::unproject(const Magnum::Vector2i &window_position,
                            Magnum::Float depth) const {
  const Vector2i view_position{window_position.x(),
                               _frame_size.y() - window_position.y() - 1};
  const Vector3 in{2 * Vector2{view_position} / Vector2{_frame_size} -
                       Vector2{1.0f},
                   depth * 2.0f - 1.0f};
  /*
  Use the following to get global coordinates instead of camera-relative:

  (_camera_parent->absoluteTransformationMatrix()*_camera->projectionMatrix().inverted()).transformPoint(in)
  */
  return _camera->projectionMatrix().inverted().transformPoint(in);
}

Float CameraController::depth_at(const Vector2i &windowPosition) {
  /* First scale the position from being relative to window size to being
     relative to framebuffer size as those two can be different on HiDPI
     systems */
  const Vector2i position = windowPosition * Vector2{_app->framebufferSize()} /
                            Vector2{_app->windowSize()};
  const Vector2i fbPosition{position.x(),
                            GL::defaultFramebuffer.viewport().sizeY() -
                                position.y() - 1};

  GL::defaultFramebuffer.mapForRead(
      GL::DefaultFramebuffer::ReadAttachment::Front);
  auto data = GL::defaultFramebuffer.read(
      Range2Di::fromSize(fbPosition, Vector2i{1}).padded(Vector2i{2}),
      {GL::PixelFormat::DepthComponent, GL::PixelType::Float});

  /* TODO: change to just Math::min<Float>(data.pixels<Float>() when the
     batch functions in Math can handle 2D views */
  return Math::min<Float>(data.pixels<Float>().asContiguous());
}

} // namespace pc::camera
