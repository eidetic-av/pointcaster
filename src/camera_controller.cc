#include "camera_controller.h"
#include "math.h"
#include "log.h"
#include <numbers>
#include <algorithm>

#include <Magnum/Image.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/Math/FunctionsBatch.h>
#include <Magnum/Math/Vector3.h>

using namespace Magnum;

std::atomic<uint> CameraController::count = 0;

CameraController::CameraController(Magnum::Platform::Application *app,
				   Object3D &object)
    : _app(app), Object3D{&object} {

  _yawObject = new Object3D{this};
  _pitchObject = new Object3D{_yawObject};

  // set initial orientation
  _yawObject->rotate(Rad(std::numbers::pi / 2), Vector3::yAxis(1));
  _pitchObject->rotate(Rad(-std::numbers::pi / 6), Vector3::zAxis(1));

  _cameraObject = new Object3D{_pitchObject};
  _cameraObject->setTransformation(
      Matrix4::lookAt({7, 7, 0}, Vector3{}, Vector3::yAxis(1)));

  _camera = std::make_unique<SceneGraph::Camera3D>(*_cameraObject);
  _camera->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::Extend);

  // default
  _camera->setProjectionMatrix(
      Matrix4::perspectiveProjection(Deg(45.0f), 4.0f / 3.0f, 0.001f, 200.0f));

  name = "camera_" + std::to_string(++CameraController::count);
}

CameraController::~CameraController() {
  CameraController::count--;
}

void CameraController::setupFramebuffer(Vector2i frame_size) {
  if (frame_size == _frame_size) return;
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

bool _is_dz_started = false;
float _init_height_at_distance = 0.0f;


// TODO
Matrix4 CameraController::make_projection_matrix() {
  auto frustum_height_at_distance = [](float distance, float fov) {
    return 2.0f * distance * std::tan(fov * 0.5f * Math::Constants<float>::pi() / 180.0f);
  };

  auto fov_for_height_and_distance = [](float height, float distance) {
    return 2.0f * std::atan(height * 0.5f / distance) * 180.0f / Math::Constants<float>::pi();
  };

  const auto focal_point = unproject(_frame_size / 2, depth_at(_frame_size / 2));
  auto camera_location = _cameraObject->transformation().translation();
  pc::log.info("camera_location: %f, %f, %f", camera_location.x(), camera_location.y(), camera_location.z());
  const auto target_distance = (camera_location - focal_point).length();

  if (!_is_dz_started) {
    _init_height_at_distance = frustum_height_at_distance(target_distance, _perspective_value);
    _is_dz_started = true;
  }

  auto fov =
    pc::math::remap(0.0f, 1.0f, 0.01f, 90.0f - 0.01f, _perspective_value, true);

  auto height = frustum_height_at_distance(target_distance, fov);
  auto new_fov = fov_for_height_and_distance(height, target_distance);

  // Compute new distance to maintain the initial frustum height.
  // auto new_distance = _init_height_at_distance / (2.0f * Math::tan(Deg(new_fov * 0.5f) * Math::Constants<float>::pi() / 180.0f));
  // Compute the new camera position to move towards or away from the subject as FOV changes.
  auto direction = (camera_location - focal_point).normalized();
  // camera_location = focal_point + direction * -new_distance;

  // Update the camera's position.
  // auto transform = Matrix4::from(_cameraObject->transformation().rotation(),
  // 				 camera_location);
  // _cameraObject->setTransformation(transform);

  // return Matrix4::perspectiveProjection(Deg(new_fov), 4.0f / 3.0f, 0.001f,
  return Matrix4::perspectiveProjection(
      Deg(new_fov), _frame_size.x() / _frame_size.y(), 0.001f, 200.0f);
}

CameraController &CameraController::rotate(const Magnum::Vector2i &shift) {
  Vector2 s = Vector2{shift} * _rotate_speed;
  _yawObject->rotate(Rad(s.x()), Vector3::yAxis(1));
  _pitchObject->rotate(Rad(s.y()), Vector3::zAxis(1));
  return *this;
}

CameraController &CameraController::move(
    Magnum::Platform::Sdl2Application::MouseMoveEvent &event) {
  const auto frame_centre = _frame_size / 2;
  const auto centre_depth = depth_at(frame_centre);
  const Vector3 p = unproject(event.position(), centre_depth);
  const auto movement =
      Vector3{(float)event.relativePosition().x() * _move_speed.x(),
	      (float)event.relativePosition().y() * _move_speed.y(), 0};
  _cameraObject->translateLocal(movement);
  return *this;
}

CameraController &CameraController::dolly(
    Magnum::Platform::Sdl2Application::MouseScrollEvent &event) {
  const auto delta = event.offset().y();
  const auto frame_centre = _frame_size / 2;
  const auto centre_depth = depth_at(frame_centre);
  const auto focal_point = unproject(frame_centre, centre_depth);
  _cameraObject->translateLocal(focal_point * delta * 0.01f);

  // auto speed_x = math::remap(0.0f, 1.0f, -0.01f, -0.0035f, _perspective_value);
  // auto speed_y = math::remap(0.0f, 1.0f, 0.0035f, 0.0035f, _perspective_value);
  // _move_speed = Vector2(speed_x, speed_y);

  return *this;
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

CameraController &CameraController::setSpeed(const Magnum::Vector2 &speed) {
  _move_speed = speed;
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

  (_cameraObject->absoluteTransformationMatrix()*_camera->projectionMatrix().inverted()).transformPoint(in)
  */
  return _camera->projectionMatrix().inverted().transformPoint(in);
}

Float CameraController::depth_at(const Vector2i& windowPosition) {
    /* First scale the position from being relative to window size to being
       relative to framebuffer size as those two can be different on HiDPI
       systems */
    const Vector2i position = windowPosition*Vector2{_app->framebufferSize()}/Vector2{_app->windowSize()};
    const Vector2i fbPosition{position.x(), GL::defaultFramebuffer.viewport().sizeY() - position.y() - 1};

    GL::defaultFramebuffer.mapForRead(GL::DefaultFramebuffer::ReadAttachment::Front);
    auto data = GL::defaultFramebuffer.read(
        Range2Di::fromSize(fbPosition, Vector2i{1}).padded(Vector2i{2}),
        {GL::PixelFormat::DepthComponent, GL::PixelType::Float});

    /* TODO: change to just Math::min<Float>(data.pixels<Float>() when the
       batch functions in Math can handle 2D views */
    return Math::min<Float>(data.pixels<Float>().asContiguous());
}
